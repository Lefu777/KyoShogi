GPP = C:\MyLib\x86_64-13.2.0-release-posix-seh-msvcrt-rt_v11-rev0\mingw64\bin\g++
# GPP = g++

CPPSOURCES = \
	usi_parser.cpp\
	uct_random_player.cpp\
	uct_node.cpp\
	parallel_uct_node.cpp\
	parallel_puct_player.cpp\
	main.cpp\
	gensfen_parser.cpp\
	cshogi.cpp\
	utils/MT.cpp\
	utils/util.cpp\
	test/read_teacher_test.cpp\
	nn/nn_tensorrt.cpp\
	dataloader_cpp/dataloader.cpp\
	cshogi_cpp/bitboard.cpp\
	cshogi_cpp/book.cpp\
	cshogi_cpp/common.cpp\
	cshogi_cpp/dfpn.cpp\
	cshogi_cpp/generateMoves.cpp\
	cshogi_cpp/hand.cpp\
	cshogi_cpp/init.cpp\
	cshogi_cpp/mate.cpp\
	cshogi_cpp/move.cpp\
	cshogi_cpp/mt64bit.cpp\
	cshogi_cpp/piece.cpp\
	cshogi_cpp/position.cpp\
	cshogi_cpp/search.cpp\
	cshogi_cpp/square.cpp\
	cshogi_cpp/usi.cpp\
	mate/dfpn_dlshogi.cpp\
	mate/my_dfpn_parallel.cpp\
	mate/my_dfpn_parallel2.cpp\
	mate/my_dfpn_parallel3.cpp\
	mate/my_dfpn_parallel4.cpp

# YET
# -flto
# -fivopts
# -funswitch-loops
# -fno-ira-share-spill-slots
# -fno-ira-share-save-slots
# -fdevirtualize-at-ltrans
# -fdevirtualize-speculatively
# -fbranch-target-load-optimize2
# -fbtr-bb-exclusive
# -fstdarg-opt
CPPFLAGS = -w -O3 -std=c++17 -flto=auto -mtune=native -march=native -mfpmath=both\
	-pipe -funroll-loops -ffast-math -fforce-addr -fdevirtualize-speculatively
CPPFLAGS_PARA = -ftree-parallelize-loops=2

INCLUDE_DIR = -I "." -I "cshogi_cpp" -I "utils" -I "nn" -I "dataloader_cpp" -I "test" -I "mate"\
	-I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include"\
	-I "C:\MyLib\TensorRT\TensorRT-8.5.1.7.Windows10.x86_64.cuda-11.8.cudnn8.6\TensorRT-8.5.1.7\include"
LIB_DIR = -L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64"\
-L "C:\MyLib\TensorRT\TensorRT-8.5.1.7.Windows10.x86_64.cuda-11.8.cudnn8.6\TensorRT-8.5.1.7\lib"

# https://qiita.com/argama147/items/2f636a2f4fd76f6ce130
LIB_CPP = -lnvinfer -lnvparsers -lnvonnxparser -lcudart -lkernel32\
	-luser32 -lgdi32 -lwinspool -lcomdlg32 -ladvapi32 -lshell32 -lole32 -loleaut32 -luuid -lodbc32 -lodbccp32\
	-lstdc++fs -static

DEFAULT_MACRO = -D HAVE_SSE4 -D HAVE_SSE42 -D HAVE_BMI2 -D HAVE_AVX2 -D NDEBUG

TARGET = cshogi_test1

# ================================================================================
# normal
# ================================================================================
normal: $(CPPSOURCES)
	$(GPP) $(CPPSOURCES) $(CPPFLAGS) $(INCLUDE_DIR) $(LIB_DIR) $(LIB_CPP) $(DEFAULT_MACRO) -o $(TARGET)_$@

# 変わらないor遅い
normalpara: $(CPPSOURCES)
	$(GPP) $(CPPSOURCES) $(CPPFLAGS) $(INCLUDE_DIR) $(LIB_DIR) $(LIB_CPP) $(DEFAULT_MACRO) $(CPPFLAGS_PARA) -o $(TARGET)_$@


engine: $(CPPSOURCES)
	$(GPP) $(CPPSOURCES) $(CPPFLAGS) $(INCLUDE_DIR) $(LIB_DIR) $(LIB_CPP) -D USI_ENGINE -D PARALLEL_PUCT_PLAYER -D FEATURE_V1 $(DEFAULT_MACRO) -o $(TARGET)_$@
	
# https://qiita.com/xxx-zamagi-xxx/items/9af4fc97c957a0c224a8
# test なるフォルダが存在するからか、make test だとだめ。
# unit test の意味合いでutest とした。
utest: $(CPPSOURCES)
	$(GPP) $(CPPSOURCES) $(CPPFLAGS) $(INCLUDE_DIR) $(LIB_DIR) $(LIB_CPP) -D TEST -D MOVE_GENERATION_FESTIVAL $(DEFAULT_MACRO) -o $(TARGET)_$@

# ================================================================================
# by obj
# ================================================================================
# CPPOBJS = $(addprefix build/gcc/obj/,$(CPPSOURCES:.cpp=.o))
CPPOBJS = $(CPPSOURCES:.cpp=.o)
# print: $(CPPOBJS)
# 	echo $(CPPOBJS)

# .cpp.o :
# 	$(GPP) $(CPPFLAGS) $(INCLUDE_DIR) -D TEST -D MOVE_GENERATION_FESTIVAL -D DEFAULT_MACRO -c $(<) -o $(<:.cpp=.o)

# # objct file を明示的に生成してコンパイル
# # .h にある関数定義の変化に気づかない模様。
# otest: $(CPPOBJS)
# 	$(GPP) $(CPPFLAGS) $(CPPOBJS) $(LIB_DIR) $(LIB_CPP) -D TEST -D MOVE_GENERATION_FESTIVAL -D DEFAULT_MACRO -o $(TARGET)_$@

# ================================================================================
# PGO
# ================================================================================
# generate profile
normalgenp: $(SOURCES_C)
	$(GCC) $(SOURCES_C) $(CFLAGS) $(CFLAGS_PARA) -fprofile-generate $(CLIBS) -o main

# ues profile
normalusep: $(SOURCES_C)
	$(GCC) $(SOURCES_C) $(CFLAGS) $(CFLAGS_PARA)-fprofile-use $(CLIBS) -o main

# pgo を一括で実行
normalpgo:
	$(MAKE) normalgenp
	./main learn
	$(MAKE) normalusep

# https://gcc.gnu.org/onlinedocs/gcc-8.1.0/gcc/
# https://gcc.gnu.org/onlinedocs/gcc-8.1.0/gcc/Option-Summary.html#Option-Summary
#     : 全option が載ってるのかね。
# https://gcc.gnu.org/onlinedocs/gcc-8.1.0/gcc/Directory-Options.html#Directory-Options
#     : -iquote
#         : #include "file" に適用
#     : -I
#         : #include "file", #include <file> に適用
#     : -L
#         : library のsearch dir に追加
# https://cpprefjp.github.io/implementation.html
# https://gcc.gnu.org/onlinedocs/gcc-8.1.0/gcc/C-Dialect-Options.html#C-Dialect-Options
#     : -std=c++17
#         : c++17 指定
# https://stackoverflow.com/questions/1452671/disable-all-gcc-warnings
#     : -w
#         : 全ての警告を抑制
# https://stackoverflow.com/questions/72218980/gcc-v12-1-warning-about-serial-compilation
# https://gcc.gnu.org/onlinedocs/gcc-13.2.0/gcc/Optimize-Options.html
#     : flto の仕様が変わった。-flto=auto 
# https://zenn.dev/keitean/articles/aaef913b433677
#     : make についてめっちゃ詳しく書いてる。
# https://www.gnu.org/software/make/manual/html_node/index.html#SEC_Contents
#     : make 公式ドキュメント
# https://wakky.tech/raspberry-pi-os-make-makefile-error/
#     : インデントはタブで無ければならない
# https://gammalab.net/blog/3rgve9rdw2kt5/
#     : .cpp.o なるサフィックスルール
