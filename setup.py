import os
import shutil
import sys
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is required to build this extension.")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        build_temp = os.path.join(self.build_temp, "cmake_build")

        cmake_args = [
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release"], cwd=build_temp
        )
        ext_path = self.get_ext_fullpath(ext.name)
        ext_dir = os.path.abspath(os.path.dirname(ext_path))
        source_path = os.path.join(ext.sourcedir, "mclq", "libmclq.so")
        dest_path = os.path.join(ext_dir, "mclq", "libmclq.so")
        if source_path == dest_path:
            return
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copyfile(source_path, dest_path)


setup(
    name="mclq",
    version="0.1",
    packages=find_packages(),
    ext_modules=[CMakeExtension("mclq", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    package_data={"mclq": ["libmclq.so"]},
    include_package_data=True,
    zip_safe=False,
)
