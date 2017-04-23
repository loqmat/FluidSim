#include <iostream>
#include "fluid/graphics.hpp"

namespace Fluids {

	Shader::Shader() :
		_program(glCreateProgram()),
		vertex(NULL), 
		fragment(NULL) { ; }

	Shader::~Shader() {
		glDeleteProgram(_program);
	}

	Shader::operator uint() const {
		return _program;
	}

	void Shader::__check_shader(uint shader) {
		int done = 0;
		glGetShaderiv( shader, GL_COMPILE_STATUS, &done);

		if ( done == GL_FALSE ) {
			int maxlen = 0;
			glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &maxlen );

			char* infolog = new char[maxlen+1];
			glGetShaderInfoLog( shader, maxlen, &maxlen, infolog );
			glDeleteShader( shader );

			std::cerr << infolog << std::endl;

			delete[] infolog;

			throw "Error in Shader";
		}
	}
	void Shader::__check_program(uint handle) {
		int is_linked;
		glGetProgramiv(handle, GL_LINK_STATUS, &is_linked);
		if ( is_linked == GL_FALSE ) {
			int maxlen = 0;
			glGetProgramiv(handle, GL_INFO_LOG_LENGTH, &maxlen);

			char* infolog = new char[maxlen+1];
			glGetProgramInfoLog(handle, maxlen, &maxlen, infolog);
			glDeleteProgram(handle);

			std::cerr << infolog << std::endl;

			delete[] infolog;

			throw "Error in Program";
		}
	}

	void Shader::link() {
		uint vshader = 0, fshader = 0;

		vshader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vshader, 1, &vertex, NULL);
		glCompileShader(vshader);

		__check_shader(vshader);

		glAttachShader(_program, vshader);

		fshader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fshader, 1, &fragment, NULL);
		glCompileShader(fshader);

		__check_shader(fshader);

		glAttachShader(_program, fshader);

		glLinkProgram(_program);
		
		glDeleteShader( vshader );
		glDeleteShader( fshader );

		__check_program(_program);
	}

}