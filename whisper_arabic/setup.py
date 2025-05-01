from setuptools import setup, find_packages

setup(
    name="whisper_arabic",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.20.0",
        "soundfile>=0.10.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Arabic Speech Recognition using fine-tuned Whisper model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/whisper_arabic",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)