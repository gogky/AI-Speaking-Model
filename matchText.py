import Levenshtein
import difflib

def text_match_score(script, transcript):
    return Levenshtein.ratio(script, transcript)

def text_differences(a: str, b: str):
    """
    打印出字符串 a 与 b 的字符级别差异。
    标记：
      - 以 '-' 开头的行表示在 a 中但不在 b 中（删除）
      - 以 '+' 开头的行表示在 b 中但不在 a 中（新增）
      - 以 ' ' 开头的行表示相同
    """

    diff = difflib.ndiff(a, b)
    show_text = ""
    state = ''
    for line in diff:
        pure_s = line[2:]
        new_state = ''
        if line.startswith('- '):
            new_state = '-'
        elif line.startswith('+ '):
            new_state = '+'
        else:
            new_state = ''
        prefix = ''
        postfix = ''
        if state == '':
            if new_state == '-':
                prefix = '[-'
            elif new_state == '+':
                prefix = '[+'
        elif state == '-':
            if new_state == '+':
                prefix = '][+'
            elif new_state == '':
                prefix = ']'
        elif state == '+':
            if new_state == '-':
                prefix = '][-'
            elif new_state == '':
                prefix = ']'
        state = new_state
        show_text += prefix + pure_s + postfix
        # print(pure_s, end='')
    if state != '':
        show_text += ']'
    return show_text

if __name__ == "__main__":
    # 示例文本
    script = "This is a test script. hallo world"
    transcript = "This is a test transcript."

    score = text_match_score(script, transcript)
    print(f"Text match score: {score:.2f}")

    print("Differences:")
    print(text_differences(script, transcript))
