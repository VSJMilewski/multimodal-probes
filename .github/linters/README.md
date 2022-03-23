The configurations for the super-linter linters are changed, because they triggered incorrectly. 
Flake8 does not always play nice with black, and it missinterpretted some escaped characters as
regular expression.

jscpd started recognising method arguments and docstring as copies, which doesn't make sense ofcourse.