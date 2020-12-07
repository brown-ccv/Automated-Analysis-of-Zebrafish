import pytest
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../src'))

import read_data

images = read_data('example_images/96_well_plates/IMG_000%1d.JPG') # loading images from testing

def test_read(images):

    ret, frame, frame_no = images.read()

    assert ret == True
    assert frame_no == 1

    ret, frame, frame_no = images.read()

    assert ret == True
    assert frame_no == 2

def test_reset(images):

    images.reset()
    _, _, frame_no = images.read()

    assert frame_no == 1

def test_get_shape(images):

    shape = images.get_shape()

    assert shape == (5184, 3456, 3)

def test_get_total_frames(images):

    total_frames = images.get_total_frames()

    assert total_frames == 9