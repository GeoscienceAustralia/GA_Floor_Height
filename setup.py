from setuptools import setup, find_packages

setup(
    name="floorheight",
    version="1.0",
    description="First floor height estimation",
    author="Lavender Liu",
    author_email="lliu@frontiersi.com.au",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pillow",
        "pydantic",
        "httpx"
    ],
)
