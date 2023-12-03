#include "read_teacher_test.hpp"

static constexpr int SFEN_BUF_SIZE = 1024;
static constexpr int PATH_BUF_SIZE = 1024;

// 自作dtype型の配列に突っ込む方法。cshogi のファイル検索したら出てくるやろ。
int read_teacher_test0(std::string file_path) {
	std::vector<teacher_t> teachers;
	FILE* file;
	if ((fopen_s(&file, file_path.c_str(), "rb")) != 0) {
		// https://qiita.com/izuki_y/items/26bf20c4b3b3750ab7a7
		std::cout << "Error: failed to open." << std::endl;
		return 0;
	}
	size_t sfen_size;
	char sfen_char[SFEN_BUF_SIZE];
	int move;
	int ply;
	float value;
	float result;
	// https://oshiete.goo.ne.jp/qa/7050751.html
	while (true) {
		auto rsize = fread(&sfen_size, sizeof(size_t), 1, file);
		if (rsize != 1) {    // 読み込みエラー or 終端
			break;
		}
		if (sfen_size > SFEN_BUF_SIZE) {
			std::cout << "Error: run out of buffer size (sfen_size = " << sfen_size << " > SFEN_BUF_SIZE)" << std::endl;
			return 0;
		}
		fread(sfen_char, sizeof(char), sfen_size, file);
		fread(&move, sizeof(int), 1, file);
		fread(&ply, sizeof(int), 1, file);
		fread(&value, sizeof(float), 1, file);
		fread(&result, sizeof(float), 1, file);
		std::string sfen_str(sfen_char, sfen_size);
		teachers.emplace_back(sfen_str, move, ply, value, result);
	}

	if (!feof(file)) {
		std::cout << "Error: failed to parse " << file_path << std::endl;
		return 0;
	}

	system("pause");

	auto teachers_size = teachers.size();
	int i = 0;
	for (const auto& e : teachers) {
		std::cout
			<< "[" << i << "]"
			<< "[" << e.sfen << "]"
			<< "[" << e.move << "]"
			<< "[" << __move_to_usi(e.move) << "]"
			<< "[" << e.ply << "]"
			<< "[" << e.value << "]"
			<< "[" << e.result << "]"
			<< std::endl;
		++i;
	}
	std::cout << "teachers_size = " << teachers_size << std::endl;
	fclose(file);

	return 1;
}

// https://learn.microsoft.com/ja-jp/windows/win32/fileio/listing-the-files-in-a-directory
// @arg dir: ex> dir = "./folder_name"
int read_teacher_test1(std::string dir) {
	std::vector<std::string> file_names;
	using namespace std::filesystem;
	directory_iterator iter(dir), end;
	std::error_code err;

	for (; iter != end && !err; iter.increment(err)) {
		const directory_entry entry = *iter;

		file_names.push_back(entry.path().string());
		const std::string& file_name = file_names.back();
		std::cout << "info: [" << file_name << "][" << (file_name.substr(-4)) << "]" << std::endl;;
		//std::cout << "info: [" << file_name << "][" << (file_name.substr(file_name.size() - 4)) << "][" << (file_name.substr(file_name.size() - 4) == ".bin" ? "true" : "false") << "]" << std::endl;;
	}

	/* エラー処理 */
	if (err) {
		std::cout << err.value() << std::endl;
		std::cout << err.message() << std::endl;
		return 0;
	}
	return 1;
}