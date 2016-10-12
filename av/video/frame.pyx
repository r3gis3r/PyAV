from av.bytesource cimport ByteSource, bytesource
from av.utils cimport err_check
from av.video.format cimport get_video_format, VideoFormat
from av.video.plane cimport VideoPlane
from av.packet cimport Packet
cimport numpy as np
import numpy as np
cimport cython

from libc.string cimport strncmp, strcpy, memcpy


colorspace_flags = {
    'itu709': lib.SWS_CS_ITU709,
    'fcc': lib.SWS_CS_FCC,
    'itu601': lib.SWS_CS_ITU601,
    'itu624': lib.SWS_CS_SMPTE170M,
    'smpte240': lib.SWS_CS_SMPTE240M,
    'default': lib.SWS_CS_DEFAULT,
    None: lib.SWS_CS_DEFAULT,
}

resize_method_flags = {
    'fast_bilinear': lib.SWS_FAST_BILINEAR,
    'bilinear': lib.SWS_BILINEAR,
    'bicubic': lib.SWS_BICUBIC,
    'area': lib.SWS_AREA,
    None: lib.SWS_BILINEAR,
}

cdef object _cinit_bypass_sentinel

cdef VideoFrame alloc_video_frame():
    """Get a mostly uninitialized VideoFrame.

    You MUST call VideoFrame._init(...) or VideoFrame._init_properties()
    before exposing to the user.

    """
    return VideoFrame.__new__(VideoFrame, _cinit_bypass_sentinel)


cdef class VideoFrame(Frame):

    """A frame of video.

    >>> frame = VideoFrame(1920, 1080, 'rgb24')

    """

    def __cinit__(self, width=0, height=0, format='yuv420p', buffer_size=0):

        if width is _cinit_bypass_sentinel:
            return

        cdef lib.AVPixelFormat c_format = lib.av_get_pix_fmt(format)
        if c_format < 0:
            raise ValueError('invalid format %r' % format)

        self._init(c_format, width, height, buffer_size)

    cdef _init(self, lib.AVPixelFormat format, unsigned int width, unsigned int height, int buffer_size):
        cdef int requested_buffer_size = buffer_size
        with nogil:
            self.ptr.width = width
            self.ptr.height = height
            self.ptr.format = format

            if width and height:

                # Cleanup the old buffer.
                lib.av_freep(&self._buffer)
                # Get a new one.
                if buffer_size == 0 or format != lib.AV_PIX_FMT_YUV420P:
                    buffer_size = lib.avpicture_get_size(format, width, height)
                    with gil: err_check(buffer_size)
                else:
                    self.ptr.linesize[0] = buffer_size * 2 / (3 * height)
                    self.ptr.linesize[1] = self.ptr.linesize[2] = buffer_size * 1 / (3 * height)


                self._buffer = <uint8_t *>lib.av_malloc(buffer_size)

                if not self._buffer:
                    with gil: raise MemoryError("cannot allocate VideoFrame buffer")
                # Attach the AVPicture to our buffer.
                if requested_buffer_size == 0 or format != lib.AV_PIX_FMT_YUV420P:
                    lib.avpicture_fill(
                            <lib.AVPicture *>self.ptr,
                            self._buffer,
                            format,
                            width,
                            height
                    )
                else:
                    # TODO : condition ffmpeg available version
                    lib.av_image_fill_pointers(
                        self.ptr.data,
                        format,
                        height,
                        self._buffer,
                        self.ptr.linesize,
                    )


        self._init_properties()

    cdef int _max_plane_count(self):
        return self.format.ptr.nb_components

    cdef _init_properties(self):
        self.format = get_video_format(<lib.AVPixelFormat>self.ptr.format, self.ptr.width, self.ptr.height)
        self._init_planes(VideoPlane)

    def __dealloc__(self):
        lib.av_freep(&self._buffer)

    def __repr__(self):
        return '<av.%s #%d, %s %dx%d at 0x%x>' % (
            self.__class__.__name__,
            self.index,
            self.format.name,
            self.width,
            self.height,
            id(self),
        )

    def reformat(self, width=None, height=None, format=None, src_colorspace=None, dst_colorspace=None, resize_method=None):
        """reformat(width=None, height=None, format=None, src_colorspace=None, dst_colorspace=None, resize_method=None)

        Create a new :class:`VideoFrame` with the given width/height/format/colorspace.

        :param int width: New width, or ``None`` for the same width.
        :param int height: New height, or ``None`` for the same height.
        :param str format: New format, or ``None`` for the same format; see :attr:`VideoFrame.format`.
        :param str src_colorspace: Current colorspace.
        :param str dst_colorspace: Desired colorspace.

        Supported colorspaces are currently:
            - ``'itu709'``
            - ``'fcc'``
            - ``'itu601'``
            - ``'itu624'``
            - ``'smpte240'``
            - ``'default'`` or ``None``
        :param str resize_method: Desired resize method.
        Supported resize methods are currently:
            - ``'bicubic'``
            - ``'bilinear_fast'``
            - ``'bilinear'`` or ``None``

        """

        cdef VideoFormat video_format = VideoFormat(format or self.format)

        cdef int c_src_colorspace, c_dst_colorspace, c_resize_method
        try:
            c_src_colorspace = colorspace_flags[src_colorspace]
        except KeyError:
            raise ValueError('invalid src_colorspace %r' % src_colorspace)
        try:
            c_dst_colorspace = colorspace_flags[dst_colorspace]
        except KeyError:
            raise ValueError('invalid src_colorspace %r' % dst_colorspace)
        try:
            c_resize_method = resize_method_flags[resize_method]
        except KeyError:
            raise ValueError('invalid resize_method %r' % dst_colorspace)

        return self._reformat(width or self.width, height or self.height, video_format.pix_fmt, c_src_colorspace, c_dst_colorspace, c_resize_method)

    cdef _reformat(self, unsigned int width, unsigned int height, lib.AVPixelFormat dst_format,
                   int src_colorspace, int dst_colorspace,
                   int resize_method):
        if self.ptr.format < 0:
            raise ValueError("invalid source format")

        cdef lib.AVPixelFormat src_format = <lib.AVPixelFormat> self.ptr.format

        # Shortcut!
        if (
            dst_format == src_format and
            width == self.ptr.width and
            height == self.ptr.height and
            dst_colorspace == src_colorspace
        ):
            return self

        # If we don't have a SwsContextProxy, create one.
        if not self.reformatter:
            self.reformatter = VideoReformatter()

        # Try and reuse existing SwsContextProxy
        # VideoStream.decode will copy its SwsContextProxy to VideoFrame
        # So all Video frames from the same VideoStream should have the same one
        with nogil:
            self.reformatter.ptr = lib.sws_getCachedContext(
                self.reformatter.ptr,
                self.ptr.width,
                self.ptr.height,
                src_format,
                width,
                height,
                dst_format,
                resize_method,
                NULL,
                NULL,
                NULL
            )

        cdef int *inv_tbl, *tbl, *rgbTbl
        cdef int srcRange, dstRange, brightness, contrast, saturation
        cdef int ret
        with nogil:
            if src_colorspace != lib.SWS_CS_DEFAULT or dst_colorspace != lib.SWS_CS_DEFAULT:
                ret = lib.sws_getColorspaceDetails(self.reformatter.ptr, &inv_tbl, &srcRange, &tbl, &dstRange, &brightness, &contrast, &saturation)
                if not ret < 0:
                    if src_colorspace != lib.SWS_CS_DEFAULT:
                        inv_tbl = lib.sws_getCoefficients(src_colorspace)
                    if dst_colorspace != lib.SWS_CS_DEFAULT:
                        tbl = lib.sws_getCoefficients(dst_colorspace)
                    lib.sws_setColorspaceDetails(self.reformatter.ptr, inv_tbl, srcRange, tbl, dstRange, brightness, contrast, saturation)

        # Create a new VideoFrame.
        cdef VideoFrame frame = alloc_video_frame()
        frame._init(dst_format, width, height, 0)

        # Finally, scale the image.
        with nogil:
            lib.sws_scale(
                self.reformatter.ptr,
                self.ptr.data,
                self.ptr.linesize,
                0, # slice Y
                self.ptr.height,
                frame.ptr.data,
                frame.ptr.linesize,
            )

        # Copy some properties.
        frame._copy_attributes_from(self)
        return frame

    property width:
        """Width of the image, in pixels."""
        def __get__(self): return self.ptr.width

    property height:
        """Height of the image, in pixels."""
        def __get__(self): return self.ptr.height

    property key_frame:
        """Is this frame a key frame?"""
        def __get__(self): return self.ptr.key_frame

    def to_rgb(self, **kwargs):
        """Get an RGB version of this frame.

        Any ``**kwargs`` are passed to :meth:`VideoFrame.reformat`.

        >>> frame = VideoFrame(1920, 1080)
        >>> frame.format.name
        'yuv420p'
        >>> frame.to_rgb().format.name
        'rgb24'

        """
        return self.reformat(format="rgb24", **kwargs)

    def to_image(self, **kwargs):
        """Get an RGB ``PIL.Image`` of this frame.

        Any ``**kwargs`` are passed to :meth:`VideoFrame.reformat`.

        """
        from PIL import Image
        return Image.frombuffer("RGB", (self.width, self.height), self.reformat(format="rgb24", **kwargs).planes[0], "raw", "RGB", 0, 1)

    @cython.boundscheck(False)
    def to_nd_array(self, **kwargs):
        """Get a numpy array of this frame.

        Any ``**kwargs`` are passed to :meth:`VideoFrame.reformat`.

        """
        cdef np.ndarray[np.uint8_t, ndim=1] out_np_array
        cdef int yuv_plane_size
        cdef VideoFrame frame = self.reformat(**kwargs)
        # TODO: Make this more general (if we can)
        if frame.ptr.format in (lib.AV_PIX_FMT_YUV420P,):
            # Pseudo flat format for yuv, useful for serialization/deserialization made faster
            yuv_plane_size = frame.ptr.linesize[0] * frame.ptr.height
            buffer_size = yuv_plane_size * 3 / 2
            out_np_array = np.empty((buffer_size,), dtype=np.uint8)
            # out_np_array[:yuv_plane_size] = np.frombuffer(frame.planes[0], np.uint8)
            planes = frame.planes
            out_np_array[:yuv_plane_size] = planes[0] # np.frombuffer(frame.planes[0], np.uint8)
            out_np_array[yuv_plane_size:yuv_plane_size*5/4] = planes[1] # np.frombuffer(frame.planes[1], np.uint8)
            out_np_array[yuv_plane_size*5/4:] = planes[2] # np.frombuffer(frame.planes[2], np.uint8)
            return out_np_array

        if len(frame.planes) != 1:
            raise ValueError('Cannot conveniently get numpy array from multiplane frame')

        # We only suppose this convenience for a few types.
        if frame.ptr.format in (lib.AV_PIX_FMT_RGB24, lib.AV_PIX_FMT_BGR24):
            return np.frombuffer(frame.planes[0], np.uint8).reshape(frame.ptr.height, frame.ptr.width, -1)
        # TODO : express this a more performant way
        if frame.format.name in ('gray16le', 'gray16be'):
            return np.frombuffer(frame.planes[0], np.dtype('<u2')).reshape(frame.height, frame.width)

        else:
            raise ValueError("Cannot conveniently get numpy array from %s format" % frame.format.name)

    def to_qimage(self, **kwargs):
        """Get an RGB ``QImage`` of this frame.

        Any ``**kwargs`` are passed to :meth:`VideoFrame.reformat`.
        
        Returns a ``(VideoFrame, QImage)`` tuple, where the ``QImage`` references
        the data in the ``VideoFrame``.
        """
        from PyQt4.QtGui import QImage
        from sip import voidptr

        cdef VideoFrame rgb = self.reformat(format='rgb24', **kwargs)
        ptr = voidptr(<long><void*>rgb.ptr.extended_data[0])

        return rgb, QImage(ptr, rgb.ptr.width, rgb.ptr.height, QImage.Format_RGB888)

    @staticmethod
    def from_image(img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        frame = VideoFrame(img.size[0], img.size[1], 'rgb24')

        # TODO: Use the buffer protocol.
        try:
            frame.planes[0].update(img.tobytes())
        except AttributeError:
            frame.plates[0].update(img.tostring())

        return frame

    @staticmethod
    def from_ndarray(array, format='rgb24', width=None, height=None, **kwargs):

        # TODO: We could stand to be more accepting.
        assert array.dtype == 'uint8'

        cdef int yuv420_size

        if format in ('rgb24', 'bgr24', 'gray16le', 'gray16be'):
            assert array.ndim == 3
            assert array.shape[2] == 3
            frame = VideoFrame(array.shape[1], array.shape[0], format)
            frame.planes[0].update(array.reshape(-1))
        elif format in ('yuv420p',):
            frame = VideoFrame(width, height, format, array.shape[0])
            yuv_plane_size = frame.ptr.linesize[0]*frame.ptr.height
            frame.planes[0].update(array[:yuv_plane_size])
            frame.planes[1].update(array[yuv_plane_size:yuv_plane_size*5/4])
            frame.planes[2].update(array[yuv_plane_size*5/4:yuv_plane_size*3/2])
        else:
            raise ValueError("Cannot conveniently get from numpy array with %s format" % format)
        if kwargs:
            frame.set_attributes(kwargs)
        return frame

    def get_attributes(self):
        attributes = Frame.get_attributes(self)
        if self.ptr:
            attributes.update({
                "key_frame": self.ptr.key_frame,
                "width": self.ptr.width,
                "height": self.ptr.height,
            })
        return attributes

    def set_attributes(self, attributes):
        Frame.set_attributes(self, attributes)
        if self.ptr:
            if "key_frame" in attributes:
                self.ptr.key_frame = attributes["key_frame"]

    def get_subtitle_packet(self, subtitle_stream):
        # TODO : implement
        cdef int i
        for i in range(self.ptr.nb_side_data):
            if self.ptr.side_data[i].type == lib.AV_FRAME_DATA_A53_CC:
                break
        else:
            return None

        cdef Packet packet = Packet()
        lib.av_new_packet(&packet.struct, self.ptr.side_data[i].size)
# +            lavfi->sink_stream_subcc_map[sink_idx] = avctx->nb_streams;
# +            if (!(st = avformat_new_stream(avctx, NULL)))
# +                return AVERROR(ENOMEM);
# +            st->codec->codec_id = AV_CODEC_ID_EIA_608;
# +            st->codec->codec_type = AVMEDIA_TYPE_SUBTITLE;
        memcpy(packet.struct.data, self.ptr.side_data[i].data, self.ptr.side_data[i].size)
        packet.stream = subtitle_stream
        packet.struct.pts = self.ptr.pts
        # packet.struct.pos = av_frame_get_pkt_pos(self.ptr)

        return packet
