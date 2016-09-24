from av.audio.format cimport get_audio_format
from av.audio.layout cimport get_audio_layout
from av.audio.plane cimport AudioPlane
from av.utils cimport err_check


AV_TO_NUMPY_TYPE = {
    "u8": "uint",
    "s16": "int16",
    "s32": "int32",
    "flt": "float32",
    "dbl": "float64",
    "u8p": "uint",
    "s16p": "int16",
    "s32p": "int32",
    "fltp": "float32",
    "dblp": "float64",
}

NUMPY_TO_AV_TYPE = {
    # We relayout in non planar type
    numpy_t: av_t for av_t, numpy_t in AV_TO_NUMPY_TYPE.iteritems() if not av_t.endswith("p")
}

cdef object _cinit_bypass_sentinel

cdef AudioFrame alloc_audio_frame():
    """Get a mostly uninitialized AudioFrame.

    You MUST call AudioFrame._init(...) or AudioFrame._init_properties()
    before exposing to the user.

    """
    return AudioFrame.__new__(AudioFrame, _cinit_bypass_sentinel)


cdef class AudioFrame(Frame):

    """A frame of audio."""
    
    def __cinit__(self, format='s16', layout='stereo', samples=0, align=True):
        if format is _cinit_bypass_sentinel:
            return
        raise NotImplementedError()

    cdef _init(self, lib.AVSampleFormat format, uint64_t layout, unsigned int nb_samples, bint align):
        self.align = align
        self.ptr.nb_samples = nb_samples
        self.ptr.format = <int>format
        self.ptr.channel_layout = layout

        # HACK: It really sucks to do this twice.
        self._init_properties()

        cdef size_t buffer_size
        if self.layout.channels and nb_samples:
            
            # Cleanup the old buffer.
            lib.av_freep(&self._buffer)

            # Get a new one.
            self._buffer_size = err_check(lib.av_samples_get_buffer_size(
                NULL,
                len(self.layout.channels),
                nb_samples,
                format,
                align
            ))
            self._buffer = <uint8_t *>lib.av_malloc(self._buffer_size)
            if not self._buffer:
                raise MemoryError("cannot allocate AudioFrame buffer")

            # Connect the data pointers to the buffer.
            err_check(lib.avcodec_fill_audio_frame(
                self.ptr, 
                len(self.layout.channels), 
                <lib.AVSampleFormat>self.ptr.format,
                self._buffer,
                self._buffer_size,
                self.align
            ))
            
            self._init_planes(AudioPlane)

    cdef _recalc_linesize(self):
        lib.av_samples_get_buffer_size(
            self.ptr.linesize,
            len(self.layout.channels),
            self.ptr.nb_samples,
            <lib.AVSampleFormat>self.ptr.format,
            self.align
        )
        # We need to reset the buffer_size on the AudioPlane/s. This is
        # an easy, if inefficient way.
        self._init_planes(AudioPlane)

    cdef _init_properties(self):
        self.layout = get_audio_layout(0, self.ptr.channel_layout)
        self.format = get_audio_format(<lib.AVSampleFormat>self.ptr.format)
        self.nb_channels = lib.av_get_channel_layout_nb_channels(self.ptr.channel_layout)
        self.nb_planes = self.nb_channels if lib.av_sample_fmt_is_planar(<lib.AVSampleFormat>self.ptr.format) else 1
        self._init_planes(AudioPlane)

    def __dealloc__(self):
        lib.av_freep(&self._buffer)
    
    def __repr__(self):
        return '<av.%s %d, %d samples at %dHz, %s, %s at 0x%x>' % (
            self.__class__.__name__,
            self.index,
            self.samples,
            self.rate,
            self.layout.name,
            self.format.name,
            id(self),
        )
    
    property samples:
        """Number of audio samples (per channel) """
        def __get__(self):
            return self.ptr.nb_samples
    
    property rate:
        """Sample rate of the audio data. """
        def __get__(self):
            return self.ptr.sample_rate

    def to_nd_array(self, **kwargs):
        """Get a numpy array of this frame.

        Any ``**kwargs`` are ignored for now

        """
        import numpy as np
        dtype = AV_TO_NUMPY_TYPE[self.format.name]
        if self.format.name.endswith("p"):
            # Planar format result in data on separate planes
            full_frame = np.zeros((self.samples, len(self.layout.channels)), dtype=dtype)
            for channel, plane in enumerate(self.planes):
                full_frame[:, channel] = np.frombuffer(plane, dtype=dtype)
        else:
            # Non planar has data interleaved : TODO : deinterleave with numpy - reshape is not as simple
            full_frame = np.frombuffer(self.planes[0], dtype=dtype).reshape(self.samples, len(self.layout.channels))
            # full_frame[:, channel] = (np.ndarray(shape=(self.samples,), dtype=dtype,
            #                                      buffer=plane.to_bytes()))
        return full_frame
        

    @staticmethod
    def from_ndarray(array):
        cdef AudioFrame frame = alloc_audio_frame()
        import numpy as np

        # Raise a KeyError if not good
        format = lib.av_get_sample_fmt(NUMPY_TO_AV_TYPE[array.dtype.name])

        frame._init(
            format,
            lib.av_get_default_channel_layout(array.shape[1]),
            array.shape[0],
            1, # Align?
        )
        frame.planes[0].update(array.reshape(-1))

        return frame

    def get_attributes(self):
        attributes = Frame.get_attributes(self)
        attributes.update({
            "rate": self.rate,
        })
        return attributes

    def set_attributes(self, attributes):
        Frame.set_attributes(self, attributes)
        if self.ptr:
            if "rate" in attributes:
                self.ptr.sample_rate = attributes["rate"]
