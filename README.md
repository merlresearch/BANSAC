<!--
Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# BANSAC: A dynamic BAyesian Network for adaptive SAmple Consensus

<b>IEEE/CVF International Conference on Computer Vision (ICCV), 2023</b>


**Authors:**<br>
Valter Piedade (Instituto Superior Tecnico, Lisboa)<br>
Pedro Miraldo (Mitsubishi Electric Research)<br>


Aug 21, 2023

<img src="figs/plot_intro.jpg" alt="drawing" width="500"/>

<br>

**Summary:**

Implementation of BANSAC, a new guided sampling process for RANSAC. Previous methods either assume no prior information about the inlier/outlier classification of data points or use some previously computed scores in the sampling. We derive a dynamic Bayesian network that updates individual data points’ inlier scores while iterating RANSAC. At each iteration, we apply weighted sampling using the updated scores. Our method works with or without prior data point scorings. In addition, we use the updated inlier/outlier scoring for deriving a new stopping criterion for the RANSAC loop. We test our method using three real-world datasets in different applications and obtain state-of-the-art results. Our method outperforms the baselines in accuracy while needing less computational time.


## Installation

Tested in **Ubuntu 22.04**, **Python 3.10**, and **OpenCV 4.6.0** (docker image ubuntu:jammy).

### 0. Prerequisites

Requirements:
- Anaconda - use instructions https://docs.anaconda.com/anaconda/install/index.html
- Eigen3, cmake, wget, zip, git and build-essential:
```
apt install libeigen3-dev cmake wget zip git build-essential
```


Download the repository and thirdparties
```
git clone <path_to_repository>
cd bansac
git submodule update --init --recursive
```

### 1. Create virtual environment

Configure virtual environment
```
conda env create -f environment.yaml
conda activate bansac
```

### 3. Modified version of OpenCV and Install

Clone and compile Opencv, changing the calib3d module to our modified version. Change *<path_to_anaconda>* to the path to the anaconda folder.
```
cd opencv
git apply ../bansac.patch
mkdir build && cd build
mkdir install
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=install \
    -D PYTHON3_LIBRARY=${path_to_anaconda}/envs/bansac/lib \
    -D PYTHON3_INCLUDE_DIR=${path_to_anaconda}/envs/bansac/include/python3.10 \
    -D PYTHON3_EXECUTABLE=${path_to_anaconda}/envs/bansac/bin/python3.10 \
    -D PYTHON3_PACKAGES_PATH=${path_to_anaconda}/envs/bansac/lib/python3.10/site-packages \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D HAVE_opencv_python3=ON \
    -D PYTHON_DEFAULT_EXECUTABLE=${path_to_anaconda}/envs/bansac/bin/python3.10 ..
make -j${nproc}
make install
cd ../../
```


## Datasets

For essential and fundamental matrices estimation:
```
mkdir data && cd data
wget http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-ValOnly.tar
wget http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-EF.tar
tar -xf RANSAC-Tutorial-Data-ValOnly.tar
tar -xf RANSAC-Tutorial-Data-EF.tar
ln -s RANSAC-Tutorial-Data-ValOnly/val/* ./
ln -s RANSAC-Tutorial-Data/train/* ./
cd ../
```
For homography estimation:
```
cd data
wget http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/homography.tar.gz
tar -xf homography.tar.gz
ln -s homography/* ./
cd ../
```

## Usage & Testing

Essential and Fundamental matrices estimation, ```pantheon_exterior``` sequence:
```
python3 evaluation_relative_pose_script.py --problem=essential --number_pairs=4000 --sequence=pantheon_exterior
python3 evaluation_relative_pose_script.py --problem=fundamental --number_pairs=4000 --sequence=pantheon_exterior
```

For homography matrix estimation, ```HPatches``` sequence:
```
python3 evaluation_homography_script.py --sequence=HPatches

```

## Citation

If you use the software, please cite the following:

```BibTeX
@inproceedings{piedade2023,
    author = {Valter Piedade and Pedro Miraldo},
    title = {BANSAC: A dynamic BAyesian Network for adaptive SAmple Consensus},
    booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
    year = 2023,
}
```

## Contact

Pedro Miraldo<br>
Principal Research Scientist<br>
Mitsubishi Electric Research Laboratories<br>
E-Mail: miraldo@merl.com

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:

```
Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```

Before patching, OpenCV 4.6.0 is taken without modifications from [https://github.com/opencv/opencv](https://github.com/opencv/opencv) (license included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)):
```
Copyright (C) 2000-2022, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2022, OpenCV Foundation, all rights reserved.
Copyright (C) 2008-2016, Itseez Inc., all rights reserved.
Copyright (C) 2019-2022, Xperience AI, all rights reserved.
Copyright (C) 2019-2022, Shenzhen Institute of Artificial Intelligence and
                         Robotics for Society, all rights reserved.
```

The following file:
```
utils.py
```
was taken without modification from [https://github.com/ducha-aiki/ransac-tutorial-2020-data](https://github.com/ducha-aiki/ransac-tutorial-2020-data) (license included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)):
```
Copyright (C) 2020 Google LLC, University of Victoria, Czech Technical University
```

The datasets were taken without modification from [http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/](http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/). The three files ```RANSAC-Tutorial-Data-ValOnly.tar```, ```RANSAC-Tutorial-Data-EF.tar```, and ```homography.tar.gz``` provide three datasets: HPatches, EVD, and Phototourism:

* The HPatches dataset has license included in [LICENSES/MIT.txt](LICENSES/MIT.txt):
```
Copyright (C) 2020 HPatches
```

* The EVD dataset for homography matrix estimation contains 11 images with license [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)): GRAND, PKK, FACE, GIRL, SHOP, DUM, INDEX, CAFE, FOX, CAT, and VIN. Images THERE, GRAF, ADAM, and MAG are from unknown sources.


* The Phototourism dataset contain images popular landmarks originally collected by the [Yahoo Flickr Creative Commons 100M](http://www.multimediacommons.org/) under [Creative Commons](https://creativecommons.org/) licenses.
