# Bài toán question answering Zalo
## 1. Hướng giải quyết bài toán.
   - Sử dụng các thuật toán phổ biến cho bài toán question answering để giải quyết: gồm infersent, bidaf
   - Sử dụng bert pretrained (cụ thể ở đây là bert multilingual-uncased) cho task về question answering.

## 2. Chi tiết kết quả ở các hướng.
   - Đầu tiên về mặt dữ liệu lúc đầu mình chia dữ liệu thành 85 % train, 15 % test tương ứng với gần 16k mẫu train, 
   và 2k2 mẫu test. (dữ liệu được chia theo phân bố của nhãn).
   - Với các thuật toán infersent, bidaf mình có sử dụng các embedding như glove, fasttext, hay với word segment là 
   embedding báo mới cung cấp ở github: (https://github.com/sonvx/word2vecVN), ngoài ra mình có sử dụng elmo pretrained 
   tiếng việt để extract đặc trưng từ câu từ github (ELMoForManyLangs: https://github.com/HIT-SCIR/ELMoForManyLangs)
   tuy nhiên độ chính xác cao nhất ở tập public test chỉ dừng ở hơn 55 % và mô hình có xu hướng quá fit (có thể do dữ 
   liệu ít, hoặc do bộ word embedding chưa đủ tốt). Các kết quả về độ chính xác mình đang nói về F1.
   - Với hướng thứ 2 sử dụng pretrained bert:
        + Đầu tiên thử bert pretrained multilingual-uncased ngay cho dữ liệu train test thì kết quả train tập train cao
        và trên tập test xoay xung quanh 75 - 76 % chính xác và khi submit thì được 73 - 74 % ( 1 phần cho thấy tập 
        public test và tập train cho kết quả khá tương đồng).    
        + Sau đó mình có tìm thấy 2 tập dataset mới: 1 của mailong github: 
        (https://github.com/mailong25/bert-vietnamese-question-answering/tree/master/dataset) 
        thêm khoảng 3k5 mẫu, và 1 của facebook project MLQA: (https://github.com/facebookresearch/MLQA) thêm được 5k8 mẫu. 
        Cả 2 tập này đều có format như squad nhưng tập của maillong thì format như của squad 2.0 và tập fb là squad 1.1
        (tập facebook là các câu hỏi luôn có câu trả lời).
        + Mình có thêm dữ liệu này vào train nhưng kết quả gần như không tăng. Và sau khi xem lại dữ liệu của cả 2 tập
        này một context sẽ có 4 tới 5 câu trả lời có thể có hoặc không có đáp án. Mình nghĩ nó có thể làm cho mô hình 
        không cố gắng học tổng quát hóa kiến thức chung mà cố gắng để fit đúng sự khác nhau đơn thuần giữa các tập câu 
        hỏi trong cùng một context nên làm cho mô hình không tốt nữa.
        + Sau đó mình có tìm hiểu thì biết thêm một hướng về sử dụng các mô hình bert cho cross-lingual. Có nghĩa là 
        mô hình sẽ được train với dữ liệu thuộc ngôn ngữ A nhưng mô hình sẽ được dùng cho task đó nhưng với ngôn ngữ B
        và kết quả cũng khá khả thi. Lấy cảm hứng từ bài post sau: 
        https://towardsdatascience.com/bert-based-cross-lingual-question-answering-with-deeppavlov-704242c2ac6f.        
        + Mình quyết định trước khi train bert của mình với task của zalo (chỉ xác định xem context có chứa câu trả lời
        cho câu hỏi hay không) thì mình train bert với task squad 2.0, task này yêu cầu ngoài việc xác định có câu 
        trả lời hay không thì còn có cả xác định xem câu trả lời nằm ở đâu trong context. Ngoài việc kì vọng cross 
        lingual là tập dữ liệu train trên các ngôn ngữ khác nhau có thể giúp tăng thêm độ thông tin lẫn nhau thì mình còn 
        nghĩ là điểm quan trọng cho QA là việc xác định đúng các thực thể và sự liên quan của thông tin với thực thể 
        trong câu hỏi và câu trả lời (không đơn thuần là có hay không), ở đây tập dữ liệu squad cung cấp cả 2 yếu tố này.
        Do không có tài nguyên nên mình chỉ có thể lưu check point ở 1000 step một, sử dụng script train squad trên 
        github của hugging face - transformers. Sau khi mình lấy check point ở step cuối gần 7k và có đưa ra kết quả 
        tập validation squad cũng khá ổn (link checkpoint sau tuning squad: 
        https://drive.google.com/open?id=10ZMb9pNdYeyGcREzZVfUo2iH-sqGGVX2)
        + Sau đó thì mình có thử dùng checkpoint này tuning với zalo task thì tập test có F1 khoản 79-80 % và khi submit
        thì public test đạt từ 76-77 %. (Về các trick tuning thì mình sẽ nói sau). 
        + Và do vẫn muốn sử dụng tập dữ liệu QA của tiếng việt từ hai tập dataset mình có dùng thêm ở trên. Mình 
        đã convert dữ liệu của tập fb thành format dạng squad 2.0 và ghép nó với tập tiếng việt của mailong. Tiếp tục sử 
        dụng checkpoint train từ squad 2.0. Vẫn sử dụng script training squad từ hugging face. Có tuning lại một chút về
        parameter. thì có được checkpoint ở step thứ 700 (link checkpoint qua tuning dataset việt: 
        https://drive.google.com/open?id=10w18VHVqNSZRcPJS8o7SdAzzUbNj0Of3) 
        Sử dụng checkpoint này thì tập test đạt F1 hơn 82 %. và mình submit thì được 78,5 %.
        + Và do ban đầu dang dùng 2k2 dữ liệu để validation, nên mình quyết định chia lại và chỉ dành 1k dữ liệu để 
        validation, để tăng thêm dữ liệu cho việc train mô hình, tuy nhiên điều này sẽ khiến việc pick checkpoint khó 
        khăn hơn vì 1k dữ liệu không phản ánh quá chính xác độ hiệu quả của mô hình.
        + Ngoài ra thay vì chỉ sử dụng output là pooler (token phân loại ở layer cuối của BERT) thì mình có concat
        token [CLS] ở 4 layer cuối để tăng thông tin cho việc phân loại (qua option use_pooler=True or False trong code).
        Sau khi train với last layer là concat 4 token thì mình có thử 3-4 checkpoint tốt nhất để submit thì với 1 
        checkpoint có độ chính xác trên tập test 1k này là 82.9 thì cho public score là 78.68 % và 81.24 trên tập private 
        test.
        final checkpoint submit: (https://drive.google.com/open?id=1ieY6ugamzWirwHKJ_3tdhuvqfiaSsGpj)
    
## 3. Về các trick cho tuning parameter và các hướng đã làm nhưng không khả thi
  - Về phần tuning parameter. Qua các lần thử các tham số cho việc tuning mô hình mình rút ra một vài kinh nghiệm. Với 
  phần tuning cho train squad thì mình giữ nguyên các tham số như hugging-face. Khi sử dụng checkpoint qua tuning squad 
  để tuning dataset việt thì mình set lại lr=1e-5 (vì tập dataset việt bé hơn rất nhiều squad). 
  - Các tham số khi train task zalo thì mình cũng để lr=1e-5, và batch_size=30 không quá lớn để mô hình có thể nhạy cảm 
  hơn với các sample phân loại sai và do mình đặt checkpoint save khá dày.
  
  - Về những phương án đã thử nhưng không khả thi gồm: 
    + Sử dụng ensemble với tập dataset việt hay test. Sau khi pick 10 checkpoint gần checkpoint tốt nhất mình đưa output 
    vào mô hình svm/xgboost dù độ chính xác tập test lên cao nhưng kết quả trên public test lại không tốt nên mình không
    dùng.
    + Sử dụng google translate để translate tập squad tiếng anh sang tiếng việt để làm dữ liệu train thêm, phần dữ liệu 
    thêm sau khi đưa vào để train thêm thì mô hình không tăng độ chính xác quá nhiều (mới thử với các thuật toán basic 
    chưa thử trên bert) 
    + Idea chưa kịp thử là việc thử dữ liệu translate từ squad cho mô hình bert, (dù dữ liệu squad có thể không giống về 
    format cho lắm (các context squad thường dài và câu hỏi phức tạp hơn so với bộ zalo), tuy nhiên mình nghĩ nếu có thêm 
    bộ lọc cụ thể để pick các cặp context question hợp lý thì độ chính xác có thể tăng.    

## 4. Cấu trúc project
  - module_dataset: gồm 2 thư mục:
    + dataset: chứa các output dữ liệu. Trong đó thư mục dataset_preprocess/pair_sequence/train_data chứa các dữ liệu đưa
  vào train đã được process từ raw data. file train_test_origin_1k_dev.csv  và val_origin_1k.csv (là file train cuối cùng).
  định dạng: 3 cột, cột đầu là câu hỏi, cột 2 là context và cột 3 là label với sep='\t'
    + preprocess_dataset: chứa các file để xử lý dữ liệu như convert, norm data (file handle_dataset.py). Nhiệm vụ tạo 
  dataloader cho 2 p/a là basic và bert (handle_dataloader_basic.py/handle_dataloader_bert.py). Augment translate squad 
  file (run_augment.py). Và file eda_dataset.py để xem về số lượng sample nhãn, độ dài theo mức câu/token của các dataset.
  - module_evaluate: để reference các model checkpoint, lấy file submit, và 1 phần là xem các sample bị dự đoán sai.
  các bạn có thể xem qua file wrong_validation_origin.txt (để thấy được 1 vài điểm thú vị từ những sample bị dự đoán sai)
  - module_train: gồm 2 phần chính là:
    + basic_model: chứa các implement cho 2 mô hình infersent và bidaf + code train 
    + bert_model: chứa implement cho mô hình bert mà mình chủ yếu làm. 
    + save model là thư mục cần có để save mô hình bert sau khi train. Ở đây chứa luôn log của quá trình train như mình 
    nhắc ở trên.
    + thư mục tuning_squad_viet_cpoint: chứa checkpoint để chạy train mô hình.

## 5. Cách chạy.
  - Tải về file final_checkpoint_submit.zip (link ở trên) vào thư mục module_train/final_checkpoint giải nén, sau đó 
  chạy file module_evaluate/inference_model_bert.py để ra file submit.
  - Tải về file checkpoint_after_tune_squad_viet.zip vào thư mục module_train/checkpoint_tune_squad_viet giải nén, chạy
  file module_train/main.py để train mô hình với dataset zalo.   
