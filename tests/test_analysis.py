import pytest
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../src'))

import read_data
import pandas as pd
import analysis

images = read_data('example_images/96_well_plates/IMG_000%1d.JPG')
experiment = analysis(images)

def test_detect_wells(experiment):

    wells = experiment.detect_wells(R = [72, 108])

    assert len(wells) == 384

    ref_wells = pd.read_csv('example_images/96_well_plates/ref_wells.csv')

    assert wells.equals(ref_wells)

def test_crop_wells(experiment):

    wells = experiment.detect_wells(R = [72, 108])

    well_indices, cropped_images = experiment.crop_wells(wells)

    assert cropped_images.shape == (384, 152, 152, 3)