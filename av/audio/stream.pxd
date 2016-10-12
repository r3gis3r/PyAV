cimport libav as lib

from av.audio.fifo cimport AudioFifo
from av.audio.format cimport AudioFormat
from av.audio.frame cimport AudioFrame
from av.audio.layout cimport AudioLayout
from av.audio.resampler cimport AudioResampler
from av.stream cimport Stream


cdef class AudioStream(Stream):

    cdef readonly AudioLayout layout
    cdef readonly AudioFormat format
    
    # Hold onto the frames that we will decode until we have a full one.
    cdef AudioFrame next_frame
    # Hold input timebase so that we can interprete input pts (not perfect because input might have variable timebase)
    # We could instead rescale before adding into fifo, but might introduce rounding problems
    cdef lib.AVRational _input_time_base

    # For encoding.
    cdef AudioResampler resampler
    cdef AudioFifo fifo

    cpdef encode(self, AudioFrame)
    cpdef encode_fifo(self)

    cdef _encode_fifo_frame(self, AudioFrame fifo_frame)


