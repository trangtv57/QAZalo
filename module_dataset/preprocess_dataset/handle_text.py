import string
import re
import unicodedata
# from vncorenlp import VnCoreNLP


# annotator = VnCoreNLP("../../../VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')
# make global set punctuation
set_punctuations = set(string.punctuation)
# set_punctuations.remove("_")
list_punctuations_out = ['”', '”', "›", "“"]
for e_punc in list_punctuations_out:
    set_punctuations.add(e_punc)


def get_list_syllabels(path_file):
    l_line = []
    with open(path_file, "r") as rf:
        for e_line in rf.readlines():
            l_line.append(e_line.replace("\n", ""))
    return l_line


# list_syllabels = get_list_syllabels("/home/trangtv/Documents/project/QAZalo/module_dataset/preprocess_dataset/syllable_viet.csv")


def normalize_text(text):
    text = text.replace("\n", "")
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\xa0', ' ')
    return text


def remove_multi_space(text):
    text = text.replace("\t", " ")
    text = re.sub("\s\s+", " ", text)
    # handle exception when line just all of punctuation
    if len(text) == 0:
        return text
    if text[0] == " ":
        text = text[1:]
    if len(text) == 0:
        pass
    else:
        if text[-1] == " ":
            text = text[:-1]

    return "".join(text)


def handle_punctuation_one_word(text):
    # need replace | for split field in csv file

    l_new_char = []
    for e_char in text:

        if e_char not in list(set_punctuations):
            l_new_char.append(e_char)
        else:
            l_new_char.append(" {} ".format(e_char))
    text = "".join(l_new_char)

    return text


def combine_list_text(list_text):
    n_list = []
    for e_list_text in list_text:
        n_list += e_list_text
    return n_list


def is_end_of_sentence(i, line):
    exception_list = [
        "Mr.",
        "MR.",
        "GS.",
        "Gs.",
        "PGS.",
        "Pgs.",
        "pgs.",
        "TS.",
        "Ts.",
        "T.",
        "ts.",
        "MRS.",
        "Mrs.",
        "mrs.",
        "Tp.",
        "tp.",
        "Kts.",
        "kts.",
        "BS.",
        "Bs.",
        "Co.",
        "Ths.",
        "MS.",
        "Ms.",
        "TT.",
        "TP.",
        "tp.",
        "ĐH.",
        "Corp.",
        "Dr.",
        "Prof.",
        "BT.",
        "Ltd.",
        "P.",
        "MISS.",
        "miss.",
        "TBT.",
        "Q.",
    ]
    if i == len(line)-1:
        return True

    if line[i+1] != " ":
        return False

    if i < len(line)-2 and line[i+2].islower():
        return False
    #
    # if re.search(r"^(\d+|[A-Za-z])\.", line[:i+1]):
    #     return False

    for w in exception_list:
        pattern = re.compile("%s$" % w)
        if pattern.search(line[:i+1]):
            return False

    return True


# may be last line is name of author so we should remove with some condition
def check_last_line(list_line):
    if len(list_line[-1]) < 16:
        list_line = list_line[:-1]
    return list_line


def sent_tokenize(line):
    """Do sentence tokenization by using regular expression"""
    sentences = []
    cur_pos = 0
    if not re.search(r"\.", line):
        return [line]

    for match in re.finditer(r"\.", line):
        _pos = match.start()
        end_pos = match.end()
        if is_end_of_sentence(_pos, line):
            tmpsent = line[cur_pos:end_pos]
            tmpsent = tmpsent.strip()
            cur_pos = end_pos
            sentences.append(tmpsent)

    if len(sentences) == 0:
        sentences.append(line)
    elif cur_pos < len(line)-1:
        sentences.append(line[cur_pos+1:])
    return sentences



def check_line_squad(question, document):
    list_syllabels = []
    document = handle_punctuation_one_word(document)
    document = remove_multi_space(document)
    if len(document.split(" ")) > 400:
        return False
    else:
        question = handle_punctuation_one_word(question)
        question = remove_multi_space(question)
        question_lower = question.lower()
        arr_token_lower = question_lower.split(" ")
        arr_token = question.split(" ")

        list_viet_word = (list(set_punctuations) + list_syllabels)
        for e_token in arr_token_lower:
            if e_token.isdigit() or e_token not in list_viet_word:
                return True

        count_title = 0
        for e_token in arr_token[1:]:
            if e_token.istitle():
                count_title += 1

        if count_title >= 1:
            return True
    return False


def handle_text_qa(text):
    text = normalize_text(text)
    # text = handle_punctuation_one_word(text)
    text = remove_multi_space(text)
    return text

# because pretrained embedding word2vec with segment is lower. so =>>
def handle_text_qa_with_segment(text):
    text = text.lower()
    text = normalize_text(text)

    # arr_text = annotator.tokenize(text)
    # arr_text_combine = combine_list_text(arr_text)
    # text = " ".join(arr_text_combine)
    text = handle_punctuation_one_word(text)
    text = remove_multi_space(text)
    text = text.lower()
    return text


if __name__ == '__main__':
    text = 'Vào tháng 12, Beyoncé cùng với một loạt những người nổi tiếng khác đã hợp tác và tạo ra một chiến dịch video cho "Request A Plan", một nỗ lực lưỡng đảng của một nhóm 950 thị trưởng Mỹ và những người khác được thiết kế để gây ảnh hưởng đến chính phủ liên bang. sau vụ nổ súng trường tiểu học Sandy Hook. Beyoncé trở thành đại sứ cho chiến dịch Ngày Nhân đạo Thế giới 2012 tặng bài hát "I Was Here" và video âm nhạc của nó, được quay tại Liên Hợp Quốc, cho chiến dịch. Vào năm 2013, thông báo rằng Beyoncé sẽ hợp tác với Salma Hayek và Frida Giannini trong chiến dịch "Chime for Change" của Gucci nhằm mục đích truyền bá quyền lực cho phụ nữ. Chiến dịch được phát sóng vào ngày 28 tháng 2 đã được thiết lập cho âm nhạc mới của cô. Một buổi hòa nhạc cho sự nghiệp đã diễn ra vào ngày 1 tháng 6 năm 2013 tại London và bao gồm các hoạt động khác như Ellie Goulding, Florence and the Machine, và Rita Ora. Trước buổi hòa nhạc, cô đã xuất hiện trong một video chiến dịch phát hành vào ngày 15 tháng 5 năm 2013, nơi cô cùng với Cameron Diaz, John Legend và Kylie Minogue, mô tả cảm hứng từ mẹ của họ, trong khi một số nghệ sĩ khác tôn vinh cảm hứng cá nhân từ những người phụ nữ khác , dẫn đến lời kêu gọi gửi những bức ảnh về cảm hứng của phụ nữ từ đó một lựa chọn được trình chiếu tại buổi hòa nhạc. Beyoncé nói về mẹ cô Tina Knowles rằng món quà của cô là "tìm ra những phẩm chất tốt nhất trong mỗi con người". Với sự giúp đỡ của nền tảng gây quỹ cộng đồng Catapult, khách tham quan buổi hòa nhạc có thể chọn giữa một số dự án thúc đẩy giáo dục phụ nữ và trẻ em gái. Beyoncé cũng tham gia "Miss a Meal", một chiến dịch quyên góp thực phẩm và hỗ trợ từ thiện thiện chí thông qua các cuộc đấu giá từ thiện trực tuyến tại Charitybuzz hỗ trợ tạo việc làm trên khắp châu Âu và Hoa Kỳ'
    print(handle_text_qa_with_segment(text))
