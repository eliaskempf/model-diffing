from model_diffing.content_detection import (
    contains_latex,
    contains_table,
)


class TestContainsLatex:
    def test_inline_math(self):
        assert contains_latex("The formula is $\\frac{1}{2}$ here.") is True

    def test_display_math(self):
        assert contains_latex("$$E = mc^2$$") is True

    def test_no_latex(self):
        assert contains_latex("Regular text with $5 and $10 prices.") is False

    def test_equation_environment(self):
        text = "\\begin{equation}\nx^2 + y^2 = z^2\n\\end{equation}"
        assert contains_latex(text) is True

    def test_escaped_paren(self):
        assert contains_latex("The expression \\(x^2\\) is quadratic.") is True


class TestContainsTable:
    def test_markdown_table(self):
        table = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n"
        assert contains_table(table) is True

    def test_no_table(self):
        assert contains_table("Just text, no table here.") is False

    def test_pipe_without_header_separator(self):
        assert contains_table("| not | a | table |") is False

    def test_table_with_alignment(self):
        table = "| Left | Center | Right |\n|:-----|:------:|------:|\n| a | b | c |\n"
        assert contains_table(table) is True
