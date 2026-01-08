from setuptools import setup

requirements = [
    "einops",
    "pyrender",
    "omegaconf",
    "natsort",
    "cffi",
    "pandas",
    "yacs",
    "triton==2.0.0",
    "imageio",
    "jupyter",
    "trimesh",
    "meshcat",
    "pyquaternion",
    "matplotlib",
    "opencv-python-headless==4.11.0.86",
    "pycparser==2.21",
    "numpy==1.26.4",
    "scipy==1.13.1",
    "setuptools==69.5.1",
    "wheel==0.41.2",
    "transformers==4.40.1",
    "tensorflow==2.17.0",
    "bitsandbytes==0.38.1",
    "transforms3d==0.4.2",
    "clip @ git+https://github.com/openai/CLIP.git",
]

__version__ = "0.0.1"
setup(
    name="spatial_actor",
    version=__version__,
    description="SpatialActor",
    long_description="",
    author="Hao Shi",
    author_email="shihao1895@gmail.com",
    url="",
    keywords="Embodied AI, Robotic Learning, Manipulation",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=['spatial_actor'],
    install_requires=requirements,
    extras_require={
    },
)
