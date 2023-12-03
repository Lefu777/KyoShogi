#cython: language_level=3
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string

# https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html#verbatim-c

cdef extern from "init.hpp":
    void initTable()

initTable()

cdef extern from "position.hpp":
    cdef cppclass Position:
        @staticmethod
        void initZobrist()

Position.initZobrist()

cdef extern from "cshogi.h":
    void HuffmanCodedPos_init()
    void PackedSfen_init()
    void Book_init()

HuffmanCodedPos_init()
PackedSfen_init()
Book_init()

cdef extern from "types.hpp":
    cdef cppclass Teacher
    ctypedef Teacher teacher_t

    cdef cppclass TrainingData

cdef extern from "dataloader.h":
    cdef cppclass __FastDataloader:
        __FastDataloader() except+
        __FastDataloader(const string& dir) except+
        __FastDataloader(const string& dir, const int& batch_size) except+
        __FastDataloader(
            const string& dir,
            const int& batch_size,
            const int& minimum_read_threshold
        ) except+
        __FastDataloader(
            const string& dir,
            const int& batch_size,
            const int& minimum_read_threshold,
            const bint& shuffle
        ) except+

        void finalize();

        void set_batch_size(const int& batch_size);
        void set_dir(const string& dir);
        bint is_dir_set() const;
        void print_teachers() const;

        void store_teachers[T](
            char* _ndfeatures,
            char* _ndmove_labels,
            char* _ndvalues,
            char* _ndresults,
            const int& start_idx
        );
        void store_teachers_with_idxs[T](
            char* _ndfeatures,
            char* _nd_xxxx,
            char* _ndvalues,
            char* _ndresults,
            const vector[int]& idxs
        );

        int read_files_all();

        int size[T]();
        int xor_size();

        int shuffle_files(string src_dir, string dst_dir, string base_file_name);
        int shuffle_write_one_file(string file_path, string dst_dir, string base_file_name);

        ## sequential reader関連
        void init_at_iter[T]();
        int read_files_sequential[T]();
        int store_teachers_with_sequential_reader[T](
            char* _ndfeatures,
            char* _ndmove_labels,
            char* _ndvalues,
            char* _ndresults
        );
        bint have_extra_batch[T]() const;


cdef class FastDataloader:
    cdef __FastDataloader __fdl

    def __cinit__(self, dir = None, batch_size = None, minimum_read_threshold = None, shuffle = None):
        if dir == None:
            self.__fdl = __FastDataloader()
        elif batch_size == None:
            self.__fdl = __FastDataloader(dir.encode('ascii'))
        elif minimum_read_threshold == None:
            self.__fdl = __FastDataloader(dir.encode('ascii'), batch_size)
        else:
            # assert(minimum_read_threshold != None)
            # assert(shuffle != None)
            self.__fdl = __FastDataloader(
                dir.encode('ascii'), batch_size, minimum_read_threshold, shuffle
            )

    def finalize(self):
        self.__fdl.finalize()

    def set_batch_size(self, int batch_size):
        self.__fdl.set_batch_size(batch_size)

    def set_dir(self, str dir):
        self.__fdl.set_dir(dir)

    def is_dir_set(self):
        return self.__fdl.is_dir_set()

    def print_teachers(self):
        return self.__fdl.print_teachers()

    # @arg ndfeatures
    #     : 入力特徴量
    #     : (batch_size, ...) というshape である必要がある。
    # @arg ndmove_labels: move_to_label(move) の戻り値。多クラス分類の言葉で言うと、クラス番号。
    # @arg ndvalues:
    # @arg ndresults: 結果
    def store_teachers(
        self,
        np.ndarray ndfeatures,
        np.ndarray ndmove_labels,
        np.ndarray ndvalues,
        np.ndarray ndresults,
        int start_idx
    ):
        self.__fdl.store_teachers[teacher_t](
            ndfeatures.data,
            ndmove_labels.data,
            ndvalues.data,
            ndresults.data,
            start_idx
        )

    # NOTE: python 側での多重定義は無理なので、名前を変える。
    def store_teachers_idxs(
        self,
        np.ndarray ndfeatures,
        np.ndarray ndmove_labels,
        np.ndarray ndvalues,
        np.ndarray ndresults,
        list idxs
    ):
        cdef vector[int] idxs_vec = idxs
        self.__fdl.store_teachers_with_idxs[teacher_t](
            ndfeatures.data,
            ndmove_labels.data,
            ndvalues.data,
            ndresults.data,
            idxs_vec
        )

    # hcpe, hcpe3 に対応
    def store_hcpex_idxs(
        self,
        np.ndarray ndfeatures,
        np.ndarray ndprobability,
        np.ndarray ndvalues,
        np.ndarray ndresults,
        list idxs
    ):
        cdef vector[int] idxs_vec = idxs
        self.__fdl.store_teachers_with_idxs[TrainingData](
            ndfeatures.data,
            ndprobability.data,
            ndvalues.data,
            ndresults.data,
            idxs_vec
        )

    def size_teachers(self):
        return self.__fdl.size[teacher_t]()

    def size_hcpex(self):
        return self.__fdl.size[TrainingData]()

    def xor_size(self):
        return self.__fdl.xor_size()

    def read_files_all(self):
        return self.__fdl.read_files_all()

    def read_files_sequential_teachers(self):
        return self.__fdl.read_files_sequential[teacher_t]()

    def read_files_sequential_hcpex(self):
        return self.__fdl.read_files_sequential[TrainingData]()

    def init_at_iter_teachers(self):
        self.__fdl.init_at_iter[teacher_t]()

    def init_at_iter_hcpex(self):
        self.__fdl.init_at_iter[TrainingData]()

    def store_teachers_with_sequential_reader(
        self,
        np.ndarray ndfeatures,
        np.ndarray ndmove_labels,
        np.ndarray ndvalues,
        np.ndarray ndresults
    ):
        return self.__fdl.store_teachers_with_sequential_reader[teacher_t](
            ndfeatures.data,
            ndmove_labels.data,
            ndvalues.data,
            ndresults.data
        )

    def store_hcpex_with_sequential_reader(
        self,
        np.ndarray ndfeatures,
        np.ndarray ndmove_labels,
        np.ndarray ndvalues,
        np.ndarray ndresults
    ):
        return self.__fdl.store_teachers_with_sequential_reader[TrainingData](
            ndfeatures.data,
            ndmove_labels.data,
            ndvalues.data,
            ndresults.data
        )

    def have_extra_batch_teachers(self):
        return self.__fdl.have_extra_batch[teacher_t]()

    def have_extra_batch_hcpex(self):
        return self.__fdl.have_extra_batch[TrainingData]()