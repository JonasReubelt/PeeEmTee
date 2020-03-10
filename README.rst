.. image:: https://git.km3net.de/jreubelt/peeemtee/badges/master/pipeline.svg
    :target: https://git.km3net.de/jreubelt/peeemtee/pipelines

.. image:: https://git.km3net.de/jreubelt/peeemtee/badges/master/coverage.svg
    :target: https://km3py.pages.km3net.de/jreubelt/peeemtee/coverage


Small set of classes and functions to help analyse Photomultiplier tube (PMT)
data. PMT data often consists of many waveforms that have to be analysed e.g. by
calculating the individual charges of the PMT signals and fitting a PMT response
function to the resulting distribution.

The functionalities given in this package require a certain layout of the PMT
data. Since most of the functions use fast numpy algorithms for their
calculations, the waveform data has to be a numpy array. As the data is most
likely recorded with some kind of oscilloscope with constant time sampling, we
only use the y-values (e.g. voltage drop at the input resistor of the
oscilloscope) in the array with the following shape::

    numpy.ndarray([waveform1, waveform2, waveform3, ...])
    
while the single waveform looks like::

    numpy.ndarray([y1, y2, y3, ...])

This standardised format allows the use of the very performant pre-compiled
numpy functions like::

    numpy.sum()

for the fast calculation of the waveform charges.