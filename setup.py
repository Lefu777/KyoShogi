# from distutils.core import setup, Extension
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
# 参考: https://qiita.com/neruoneru/items/6c0fc0496620d2968b57
from numpy import get_include # cimport numpyを使う場合に必要

# ライブラリのimport 時の名前は、.pyx の名前による。
# 然し、build 時に生成される .pyd の名前を変更すればもしかするとimport 時の名前を変更出来るかもしれない。
# c++のコードをpythonで動かす。
# setup(
#     name = "BitBoardCpp",
#     ext_modules = cythonize('*.pyx'),
#     include_dirs = ["C:\\Users\\ukula\\Desktop\\Programs_VSC\\Dl-reversi_cy", get_include()],
# )

# https://docs.python.org/ja/3/distutils/setupscript.html
ext_modules = [
    Extension(
        "fast_dataloader",
        sources = [
            "src/dataloader_cpp/dataloader.cpp", "src/dataloader_cy/fast_dataloader.pyx","src/utils/util.cpp","src/utils/MT.cpp",
            "src/cshogi_cpp/bitboard.cpp", "src/cshogi_cpp/common.cpp", "src/cshogi_cpp/generateMoves.cpp", "src/cshogi_cpp/hand.cpp",
            "src/cshogi_cpp/init.cpp", "src/cshogi_cpp/move.cpp","src/cshogi_cpp/mt64bit.cpp","src/cshogi_cpp/piece.cpp",
            "src/cshogi_cpp/position.cpp", "src/cshogi_cpp/search.cpp", "src/cshogi_cpp/square.cpp", "src/cshogi_cpp/usi.cpp",
            "src/cshogi_cpp/book.cpp", "src/cshogi_cpp/mate.cpp", "src/cshogi_cpp/dfpn.cpp"
        ],
        include_dirs = [
            "src/dataloader_cpp",
            "src/cshogi_cpp",
            "src/utils",
            "src",
            ".",
            get_include()
        ],
        language="c++",
        extra_compile_args = [
            "/std:c++17", "/arch:AVX2", "/Ox", "/Ob2", "/Oi", "/Ot", "/GL", "/Gy"
        ],
        extra_link_args = [
            "/LARGEADDRESSAWARE"
        ],
        # TODO: NDEBUG 入れても良いかも知らん、一旦
        define_macros = [
            ('HAVE_SSE4', None), ('HAVE_SSE42', None), ('HAVE_AVX2', None), ("HAVE_BMI2", None), ("NDEBUG", None), ("PARALLEL_PUCT_PLAYER", None)
        ]
    )
]

# https://docs.python.org/ja/3/distutils/setupscript.html
setup(
    name = "fast_dataloader",
    cmdclass = {"build_ext": build_ext},
    ext_modules= cythonize(ext_modules)
)

# python setup.py build_ext --inplace