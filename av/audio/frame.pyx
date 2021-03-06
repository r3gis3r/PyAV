from av.audio.format cimport get_audio_format
from av.audio.layout cimport get_audio_layout
from av.audio.plane cimport AudioPlane
from av.utils cimport err_check
cimport numpy as np
cimport cython

from libc.string cimport strncmp, strcpy, memcpy

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
    # numpy_t: av_t for av_t, numpy_t in AV_TO_NUMPY_TYPE.iteritems() if not av_t.endswith("p")
    # More efficient but less future prone - we should find a way to use enum values from C
    "uint": 0, # lib.AVSampleFormat.AV_SAMPLE_FMT_U8,
    "int16": 1, # lib.AVSampleFormat.AV_SAMPLE_FMT_S16,
    "int32": 2, # lib.AVSampleFormat.AV_SAMPLE_FMT_S32,
    "float32": 3, # lib.AVSampleFormat.AV_SAMPLE_FMT_FLT,
    "float64": 4 #lib.AVSampleFormat.AV_SAMPLE_FMT_DBL
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
            # if self._buffer:
            #     PyMem_Free(self._buffer)
            #     self._buffer = NULL

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
    def from_ndarray(array, **kwargs):
        cdef AudioFrame frame = alloc_audio_frame()

        # Raise a KeyError if not good
        # cdef lib.AVSampleFormat format = lib.av_get_sample_fmt(NUMPY_TO_AV_TYPE[array.dtype.name])
        cdef lib.AVSampleFormat format = <lib.AVSampleFormat>NUMPY_TO_AV_TYPE[array.dtype.name]
        cdef int samples = array.shape[0]
        cdef int channels = array.shape[1]
        cdef np.ndarray nparray = array.reshape(-1)

        frame._init(
            format,
            lib.av_get_default_channel_layout(channels),
            samples,
            1, # Align?
        )
        memcpy(<uint8_t*>frame.ptr.extended_data[0], <uint8_t*>nparray.data, nparray.shape[0] * nparray.dtype.itemsize)
        if kwargs:
            frame.set_attributes(kwargs)

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
