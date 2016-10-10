cimport libav as lib

from av.stream cimport Stream
from av.video.format cimport VideoFormat
from av.video.frame cimport VideoFrame
from av.video.reformatter cimport VideoReformatter


cdef class VideoStream(Stream):
    
    cdef readonly VideoFormat format
    cdef _build_format(self)

    # Hold onto the frames that we will decode until we have a full one.
    cdef VideoFrame next_frame
    # Should we
    cdef int _reuse_decoded_frame

    # Common reformatter shared with all frames since it is likely to get reused.
    cdef VideoReformatter reformatter

    # Size of the last frame. Used to determine if the sws_proxy should be
    # recreated.
    cdef int last_w
    cdef int last_h
    
    cdef int encoded_frame_count
    
    cpdef encode(self, VideoFrame frame=*)
