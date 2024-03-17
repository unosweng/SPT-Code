import tree_sitter
from tree_sitter import Language, Parser
import re
import os
import test_enums

path = 'data/asts/build/my-languages.so'
print(f'abspath(path): {os.path.abspath(path)}')

LANGUAGE = {
    # test_enums.LANG_GO: Language('data/asts/build/my-languages.so', 'go'),
    #         test_enums.LANG_JAVASCRIPT: Language('data/asts/build/my-languages.so', 'javascript'),
    #         test_enums.LANG_PYTHON: Language('data/asts/build/my-languages.so', 'python'),
            test_enums.LANG_JAVA: Language('build/my-languages.so', 'java'),
            # test_enums.LANG_PHP: Language('data/asts/build/my-languages.so', 'php'),
            # test_enums.LANG_RUBY: Language('data/asts/build/my-languages.so', 'ruby'),
            # test_enums.LANG_C_SHARP: Language('data/asts/build/my-languages.so', 'c_sharp')
            }

# LANGUAGE = {enums.LANG_GO: Language('build/my-languages.so', 'go'),
#             enums.LANG_JAVASCRIPT: Language('build/my-languages.so', 'javascript'),
#             enums.LANG_PYTHON: Language('build/my-languages.so', 'python'),
#             enums.LANG_JAVA: Language('build/my-languages.so', 'java'),
#             enums.LANG_PHP: Language('build/my-languages.so', 'php'),
#             enums.LANG_RUBY: Language('build/my-languages.so', 'ruby'),
#             enums.LANG_C_SHARP: Language('build/my-languages.so', 'c_sharp')}

def main():
    parser = Parser()
    lang = 'java'
    print('[DBG] BFOR: set_language')
    parser.set_language(LANGUAGE[lang])
    print('[DBG] AFTR: set_language')

if __name__ == '__main__':
    main()