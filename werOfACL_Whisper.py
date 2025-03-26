import jiwer



def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

reference_text = read_file(r"C:\Users\38644\PycharmProjects\bachelor\datasets\ACL.test.dataset\2\acl_6060\dev\ASR_result\reference\zhou.txt")
result_text = read_file(r"C:\Users\38644\PycharmProjects\bachelor\datasets\ACL.test.dataset\2\acl_6060\dev\ASR_result\ASR\zhou.txt")

wer = jiwer.wer(reference_text, result_text)

print(f'词错误率 (WER): {wer:.2%}')