#pragma once
#include <iostream>
#include <vector>
#include <string>

typedef class PeakParser {
private:
	std::vector<std::string> _tokens;
	int _peak_idx;
	int _size;

public:
	PeakParser(const std::vector<std::string>& tokens)
		: _tokens(tokens), _peak_idx(-1), _size(tokens.size())
	{

	}

	int get_peak_idx() const { return _peak_idx; }
	std::string get_peak_token() const { return _tokens[_peak_idx]; }

	bool peak_next_if_exists() {
		++_peak_idx;
		if (_peak_idx < _size) {
			return true;
		}
		return false;
	}

	void peak_next_expected_to_exist() {
		if (!peak_next_if_exists()) {
			std::cout << "Error: expected next token to exist, but it did not." << std::endl;
			throw std::runtime_error("Error");
		}
	}

	template<typename T>
	void store_peak_token(T& x) {
		const auto& token = get_peak_token();
		x = stox<T>(token);
	}
} peak_parser_t;