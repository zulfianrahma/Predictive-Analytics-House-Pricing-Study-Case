# Laporan Proyek Machine Learning  
Oleh Zulfian Rahmadiansyah

## 1. Domain Proyek

Evaluasi atau prediksi nilai harga rumah adalah topik penting bagi perusahaan pengembang properti, lembaga keuangan, dan investor dalam rangka proses negosiasi harga dan pengambilan keputusan yang tepat. Krisis hipotek dan resesi besar yang terjadi di pasar keuangan dunia selama akhir tahun 2000-an dan 2010-an mengakibatkan terjadinya penurunan ekonomi di seluruh dunia, khususnya ekonomi Amerika Serikat. Sejak kejadian tersebut, evaluasi harga rumah telah menjadi perhatian publik dan penilaian dari pakar dalam bidang properti menjadi sangat dibutuhkan oleh pembeli rumah, penjual, lembaga keuangan, dan pemerintah [1].

Tantangan utama dalam proses evaluasi harga rumah dikarenakan evaluasi tersebut merupakan proses prediksi deret waktu non-linear dalam sistem yang kompleks. Evaluasi harga rumah disebut sebagai sistem yang kompleks disebabkan mempunyai banyak faktor yang mempengaruhi meliputi faktor ekonomi (seperti rasio pendapatan, tingkat kekosongan rumah, ketimpangan pendapatan, dll), atribut rumah (seperti usia rumah, jumlah transaksi, luas tempat tinggal, tempat parkir, jumlah kamar, lokasi geografis, dll), dan kejadian krisis (seperti pandemi dan krisis ekonomi) [2].   

Penting untuk menilai harga dari suatu rumah dengan tepat untuk memaksimalkan keuntungan. Misalnya, model yang akurat dibutuhkan oleh pembeli rumah untuk mengevaluasi nilai investasinya, oleh penjual untuk mengevaluasi keuntungan penjualan, oleh kreditur untuk mengevaluasi resiko, dan oleh manajer senior serta auditor untuk mengelola portofolio aset dengan lebih baik [1].

Penggunaan teknologi sebagai solusi dalam menyelesaikan permasalahan yang dialami oleh manusia telah lama dipelajari, salah satunya adalah teknik pembelajaran mesin (*machine learning*). Teknik pembelajaran mesin telah banyak dieksplorasi untuk kasus prediksi harga. Pembelajaran mesin membuat algoritma dan model dari data, lalu menerapkannya ke data baru untuk membuat prediksi [3]. Beberapa pembelajaran mesin yang sering digunakan adalah seperti *KNN*, *Random Forest*, dan *Boosting*.

## 2. Business Understanding

Bayangkan Anda adalah pemilik perusahaan yang bergerak di bidang jual-beli rumah atau properti. Model bisnis Anda adalah perusahaan membeli rumah dari pihak pemilik rumah kemudian menjualnya kepada pihak pembeli. Perusahaan menerima keuntungan dari selisih harga jual dan harga beli dalam penjualan rumah tersebut. Tentu saja, hal pertama yang perlu dilakukan adalah menentukan berapa harga jual dan harga beli yang optimal sehingga menghasilkan keuntungan yang maksimal. 

Secara sederhana, banyak faktor yang perlu dipertimbangkan dalam proses penentuan harga jual dan harga beli dari suatu rumah. Proses evaluasi harga rumah tersebut mempunyai banyak faktor yang mempengaruhi meliputi faktor ekonomi (seperti rasio pendapatan, tingkat kekosongan rumah, ketimpangan pendapatan, dll), atribut rumah (seperti usia rumah, jumlah transaksi, luas tempat tinggal, tempat parkir, jumlah kamar, lokasi geografis, dll), dan kejadian krisis (seperti pandemi dan krisis ekonomi). Oleh sebab itu, untuk efisiensi, Anda ingin menerapkan automasi pada sistem dalam memprediksi harga rumah dengan teknik *predictive modelling*.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga rumah?
- Berapa harga rumah dengan karakteristik atau fitur tertentu?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengetahui fitur yang paling berkorelasi dengan harga rumah.
- Membuat model machine learning yang dapat memprediksi harga rumah seakurat mungkin berdasarkan fitur-fitur yang ada.

## 3. Data Understanding

Data yang digunakan adalah data harga rumah (*House Sales Prediction*) di King County, USA. Data tersebut diperoleh melalui [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction). Data tersebut merupakan data penjualan mulai dari bulan Mei 2014 sampai dengan Mei 2015. Data tersebut memuat harga jual rumah di King County, termasuk juga Seattle. Terdapat 21 kolom paramater pada data tersebut.

### Variabel-variabel pada dataset *House Sales Prediction* adalah sebagai berikut:

|  **Variabel** |                                                        **Deskripsi**                                                       | **Tipe Data** |
|:-------------:|:--------------------------------------------------------------------------------------------------------------------------:|:-------------:|
|       id      |                                             id unik dari setiap penjualan rumah                                            |    integer    |
|      date     |                                                   tanggal penjualan rumah                                                  |     object    |
|     price     |                                                    harga jual dari rumah                                                   |     float     |
|    bedrooms   |                                                        jumlah kamar                                                        |    integer    |
|   bathrooms   |                                                     jumlah kamar mandi                                                     |     float     |
|  sqft_living  |                                                luas area interior dari rumah                                               |    integer    |
|    sqft_lot   |                                                    luas area pekarangan                                                    |    integer    |
|     floors    |                                                        jumlah lantai                                                       |     float     |
|   waterfront  |                         varibel dummy untuk menentukan apakah terkait ketersediaan pemandangan laut                        |    integer    |
|      view     |                     index nilai pemandangan dari properti yang mempunyai rentang nilai dari 0 sampai 4                     |    integer    |
|   condition   |                       index nilai dari kondisi properti yang mempunyai rentang nilai dari 1 sampai 5                       |    integer    |
|     grade     | index nilai kualitas dari properti (meliputi desain dan konstruksi bangunan) yang mempunyai rentang nilai dari 1 sampai 13 |    integer    |
|   sqft_above  |                              luas area interior dari rumah yang berada di atas permukaan tanah                             |    integer    |
| sqft_basement |                             luas area interior dari rumah yang berada di bawah permukaan tanah                             |    integer    |
|    yr_built   |                                                    tahun pembuatan rumah                                                   |    integer    |
|  yr_renovated |                                                tahun terakhir renovasi rumah                                               |    integer    |
|    zipcode    |                                                       kode pos rumah                                                       |    integer    |
|      lat      |                                                          Lattitude                                                         |     float     |
|      long     |                                                          Longitude                                                         |     float     |
| sqft_living15 |                                 luas area interior dari rumah terhadap 15 tetangga terdekat                                |    integer    |
|   sqft_lot15  |                                     luas area pekarangan terhadap 15 tetangga terdekat                                     |    integer    |

 
<img width="260" alt="df info(awal)" src="https://github.com/zulfianrahma/Predictive-Analytics-House-Pricing-Study-Case/assets/97383651/b7039e0a-31a5-4ece-a92f-8baf61ec4144">

Gambar 1. Informasi terkait dataset *House Sales Prediction*

Jika dilihat dari informasi pada Gambar 1, terdapat 2163 baris data yang tersebar pada 21 kolom data. Tipe data dari dataset tersebut terdiri atas 5 tipe data *float*, 15 tipe data *int*, dan 1 tipe data *object*. Ringkasan terkait data statistik pada dataset tersebut ditunjukkan pada Tabel 1.

Tabel 1. Deksripsi statistik terkait dataset *House Sales Prediction* 
|index|id|price|bedrooms|bathrooms|sqft\_living|sqft\_lot|floors|waterfront|view|condition|grade|sqft\_above|sqft\_basement|yr\_built|yr\_renovated|zipcode|lat|long|sqft\_living15|sqft\_lot15|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|count|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|21613\.0|
|mean|4580301520\.864988|540088\.1417665294|3\.37084162309721|2\.1147573219821405|2079\.8997362698374|15106\.967565816869|1\.4943089807060566|0\.007541757275713691|0\.23430342849211122|3\.4094295100171195|7\.656873178179799|1788\.3906907879516|291\.5090454818859|1971\.0051357978994|84\.40225790033776|98077\.93980474715|47\.56005251931708|-122\.21389640494147|1986\.552491556008|12768\.455651691113|
|std|2876565571\.312057|367127\.19648269983|0\.9300618311474517|0\.770163157217742|918\.4408970468115|41420\.51151513548|0\.5399888951423463|0\.08651719772788764|0\.7663175692736122|0\.6507430463662071|1\.175458756974335|828\.0909776519169|442\.5750426774682|29\.37341080238659|401\.6792400191759|53\.505026257473084|0\.13856371024192418|0\.14082834238139408|685\.3913042527776|27304\.17963133851|
|min|1000102\.0|75000\.0|0\.0|0\.0|290\.0|520\.0|1\.0|0\.0|0\.0|1\.0|1\.0|290\.0|0\.0|1900\.0|0\.0|98001\.0|47\.1559|-122\.519|399\.0|651\.0|
|25%|2123049194\.0|321950\.0|3\.0|1\.75|1427\.0|5040\.0|1\.0|0\.0|0\.0|3\.0|7\.0|1190\.0|0\.0|1951\.0|0\.0|98033\.0|47\.471|-122\.328|1490\.0|5100\.0|
|50%|3904930410\.0|450000\.0|3\.0|2\.25|1910\.0|7618\.0|1\.5|0\.0|0\.0|3\.0|7\.0|1560\.0|0\.0|1975\.0|0\.0|98065\.0|47\.5718|-122\.23|1840\.0|7620\.0|
|75%|7308900445\.0|645000\.0|4\.0|2\.5|2550\.0|10688\.0|2\.0|0\.0|0\.0|4\.0|8\.0|2210\.0|560\.0|1997\.0|0\.0|98118\.0|47\.678|-122\.125|2360\.0|10083\.0|
|max|9900000190\.0|7700000\.0|33\.0|8\.0|13540\.0|1651359\.0|3\.5|1\.0|4\.0|5\.0|13\.0|9410\.0|4820\.0|2015\.0|2015\.0|98199\.0|47\.7776|-121\.315|6210\.0|871200\.0|


## 4. Data Preparation

Untuk meningkatkan kualitas data yang akan digunakan pada tahap modelling, dilakukan proses *data preparation* atau penyiapan data. Proses penyiapan data yang dilakukan dapat dibagi menjadi beberapa komponen, yaitu:

### 1.	Konversi data
-	konversi nilai pada variabel **view** yang awalnya bernilai numerik mulai dari 0 sampai 4 menjadi nilai kategori. Kategori yang ditentukan adalah berikut:
    -	0 = *No view*
    - 1 = *Fair*
    - 2 = *Average*
    - 3 = *Good*
    - 4 = *Excelent*
-	konversi nilai pada variabel **condition** yang awalnya bernilai numerik mulai dari 1 sampai 5 menjadi nilai kategori. Kategori yang ditentukan adalah sebagai berikut:
    - 1 = *Poor - Worn Out*
    - 2 = *Fair – Badly worn*
    - 3 = *Average*
    - 4 = *Good*
    - 5 = *Very Good*
-	konversi nilai pada variabel **grade** yang awalnya bernilai numerik mulai dari 1 sampai 13 menjadi nilai kategori yang dimasukkan dalam varibel baru bernama **grade_category**. Oleh sebab itu, variabel **grade** akan dihilangkan setelah proses konversi. Nilai kategori tersebut mengikuti aturan rentang berikut:
    - 0 – 3 = *Bad*
    - 4 – 6 = *Okay*
    - 7 – 10 = *Better*
    - 11 – 13 = *Excelent*
-	konversi nilai pada variabel **yr_renovated** yang awalnya bernilai numerik yaitu tahun renovasi menjadi nilai kategori. Untuk bangunan yang sudah direnovasi akan mendapat kategori *Yes*, sebaliknya akan mendapat kategori *No*. Nilai kategori tersebut akan dimasukkan pada variabel baru yang bernama **renovated** sehingga variabel  **yr_renovated** dihilangkan setelah proses konversi.
-	konversi nilai pada variabel **waterfront** yang awalnya bernilai numerik menjadi nilai kategori. Untuk bangunan yang mempunyai pemandangan laut atau pesisir akan mendapat kategori *Yes*, sebaliknya akan mendapat kategori *No*.
-	melakukan teknik *one-hot-encoding* pada fitur kategori yang meliputi **waterfront, view, condition, renovated** yang hasilnya dapat ditunjukkan pada Tabel 2.

Tabel 2. Lima Data Teratas pada Dataframe setelah Proses One-Hot-Encoding
|index|price|waterfront\_No|waterfront\_Yes|view\_Average|view\_Excelent|view\_Fair|view\_Good|view\_No view|condition\_Average|condition\_Fair- Badly worn|condition\_Good|condition\_Poor- Worn out|condition\_Very Good|grade\_category\_Bad|grade\_category\_Okay|grade\_category\_Better|grade\_category\_Excelent|renovated\_No|renovated\_Yes|sqft\_area|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|221900\.0|1|0|0|0|0|0|1|1|0|0|0|0|0|0|1|0|1|0|-928\.5452199193436|
|1|538000\.0|1|0|0|0|0|0|1|1|0|0|0|0|0|0|1|0|0|1|724\.0913777615001|
|2|180000\.0|1|0|0|0|0|0|1|1|0|0|0|0|0|1|0|0|1|0|-783\.3848914003222|
|3|604000\.0|1|0|0|0|0|0|1|0|0|0|0|1|0|0|1|0|1|0|-501\.54707184183906|
|4|510000\.0|1|0|0|0|0|0|1|1|0|0|0|0|0|0|1|0|1|0|-87\.97919224597405|

### 2.	Penghapusan dan reduksi data 
-	Menghilangkan variabel **"id", "date", "zipcode", "lat", dan "long"** karena data dari variabel tersebut dirasa tidak diperlukan dalam proses pengembangkan model yang dirancang.
-	Proses penghilangan *outlier* dengan metode **IQR** (*Inter Quartile Range*)
-	Menghilangkan variabel ** “bedrooms”, “bathrooms”, “sqft_lot”, “floors”, “sqft_basement”, “yr_built”, dan “sqft_lot15”** setelah dilakukan analisa pada *correlation matrix* (Gambar 7) yang dihasilkan, dimana hanya dipertahankan 3 variabel numerik yang paling berpengaruh (**sqft_living, sqft_above, dan sqft_living15**)
-	Melakukan teknik pengurangan dimensi menggunakan *Principal Component Analysis* (**PCA**) supaya data yang berkorelasi dapat disederhanakan. Teknik PCA digunakan pada variabel **sqft_living, sqft_above, dan sqft_living15**. Dalam prosesnya, diambil komponen utama pertama yang dihasilkan sehingga komponen utama kedua dan ketiga tidak diambil. Setelah proses PCA selesai, dibuat variabel baru yang bernama **sqft_area** untuk menampung komponen utama pertama dari teknik PCA. Variabel **sqft_living, sqft_above, dan sqft_living15** dihilangkan setelah variabel **sqft_area** terbentuk.

### 3.	Normalisasi data
-	dilakukan proses *normalisasi* data numerik (**sqft_area**) menggunakan fungsi normalisasi **StandarScaler()** sehingga data numerik akan mempunyai nilai rerata (*mean*) 0 dan standar devisasi (*std*) 1.
-	Setelah dataset dipisah menjadi data latih dan data uji, dilakukan proses normalisasi terlebih dahulu pada data latih dan tidak pada data uji. Tujuan perlakuan tersebut supaya tidak terjadi kebocoran informasi pada data uji selama proses pelatihan model berlangsung.
-	Setelah model telah dilatih menggunakan data uji (sudah melalui proses normalisasi), data uji dilakukan proses normalisasi supaya data menjadi lebih mudah diolah oleh algoritma.

### 4.	Pemisahan data menjadi data latih dan data uji
-	Dilakukan proses pemisahan *dataset* menjadi data latih dan data uji dengan komposisi 90:10. Pemisahan tersebut didasarkan banyaknya jumlah dataset keseluruhan sejumlah 17186 data. Dengan jumlah dataset dan komposisi pembagian tersebut, diperoleh data latih sebanyak 15467 dan data uji sebanyak 1719. Jumlah tersebut sudah dirasa cukup proporsional.



Tujuan dari proyek ini adalah melakukan proses evaluasi atau prediksi dari harga rumah sehingga harga rumah ("price") menjadi data target. Dalam dataset yang digunakan, setiap variabel di dalamnya dapat dikelompokkan menjadi fitur numerik dan fitur kategori . Hubungan korelasi antara data target dengan fitur kategori ditunjukkan mulai dari Gambar 2 sampai dengan Gambar 6. Sedangkan, korelasi antara data target dengan fitur numerik ditunjukkan pada Gambar 7.

![price vs condition](https://github.com/zulfianrahma/Predictive-Analytics-House-Pricing-Study-Case/assets/97383651/f075c40b-7145-4996-a562-8a3d7495b862)

Gambar 2. Rata - rata "price" relatif terhadap "condition"


![price vs grade_category](https://github.com/zulfianrahma/Predictive-Analytics-House-Pricing-Study-Case/assets/97383651/506e83cc-6d1d-4fe9-89bb-2ce705cc320e)

Gambar 3. Rata - rata "price" relatif terhadap "grade_category"


![price vs view](https://github.com/zulfianrahma/Predictive-Analytics-House-Pricing-Study-Case/assets/97383651/9a5f6041-36c2-49cd-a091-c82a2db40683)

Gambar 4. Rata - rata "price" relatif terhadap "view"


![price vs waterfront](https://github.com/zulfianrahma/Predictive-Analytics-House-Pricing-Study-Case/assets/97383651/fa8c8a83-54c2-4d3f-b9b6-72243772b07b)

Gambar 5. Rata - rata "price" relatif terhadap "waterfront"


![price vs renovated](https://github.com/zulfianrahma/Predictive-Analytics-House-Pricing-Study-Case/assets/97383651/c8b73514-06f5-401c-afa4-6518bc74ea2a)

Gambar 6. Rata - rata "price" relatif terhadap "renovated"

![correlation_matrix](https://github.com/zulfianrahma/Predictive-Analytics-House-Pricing-Study-Case/assets/97383651/7b0b53e0-9367-421a-a33b-a0fc60e68535)

Gambar 7. Matriks Korelasi Fitur Numerik

## 5. Modeling
Dalam proses **modelling** pada proyek yang dirancang, digunakan tiga model *machine learning* yaitu **KNNeighbors, Random Forest Regression,** dan **AdaBoost Regressor**. 

Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. Algoritma KNN pada fase pelatihan hanya menyimpan dataset dan ketika mendapatkan data baru, data tersebut akan dimasukkan ke dalam kategori yang sangat mirip dengan data baru. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). KNN bisa digunakan untuk kasus klasifikasi dan regresi [4].

![knn-1](https://github.com/zulfianrahma/Predictive-Analytics-House-Pricing-Study-Case/assets/97383651/ea982c63-80ae-45e4-b968-ede8a46fcaca)


Gambar 8. Algoritma K-Nearest-Neigbor [5]

Random Forest Regression adalah algoritma *supervised learning* yang menggunakan metode pembelajaran *ensemble* untuk proses regresi. Pembelajaran *ensemble* adalah teknik yang menggabungkan prediksi dari beberapa algoritma pembelajaran mesin untuk membuat prediksi yang lebih akurat daripada model tunggal [6].

![random forest](https://github.com/zulfianrahma/Predictive-Analytics-House-Pricing-Study-Case/assets/97383651/28607431-4c63-4d82-a396-621c8e9f9d56)

Gambar 9. Algoritma Random Forest [6]

Algoritma boosting bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma boosting muncul dari gagasan mengenai apakah algoritma yang sederhana seperti linear regression dan decision tree dapat dimodifikasi untuk dapat meningkatkan performa. AdaBoost adalah salah model dalam algoritma boosting [7].

Untuk melakukan proses *tuning* pada setiap *hyperparameter* pada masing - masing model, digunakan modul **GridSearch** untuk membantu proses menemukan nilai *hyperparameter* yang terbaik. Paramater *GridSearch* pada setiap model adalah sebagai berikut:
- **KNNeighbors**

| **_Hyperparameter_** |        **Nilai**        |
|:--------------------:|:-----------------------:|
|      n_neighbors     |       [3, 5, 7, 9]      |
|        weights       | ['uniform', 'distance'] |
|           p          |          [1, 2]         |

  - n_neighbors = jumlah tetangga
  - weights = skema bobot
  - p = *power parameter* (1 untuk Manhattan, 2 untuk Euclidean)
    
- **RandomForest**

| **_Hyperparameter_** |   **Nilai**   |
|:--------------------:|:-------------:|
|     n_estimators     |   [50, 100]   |
|       max_depth      | [None, 5, 10] |
|   min_samples_split  |     [2, 5]    |
|   min_samples_leaf   |     [1, 2]    |

  - n_estimators = jumlah node
  - max_depth = kedalaman maksimal tiap node
  - min_samples_split = jumlah minimum sampel yang diperlukan untuk membagi node internal
  - min_samples_leaf = jumlah minimum sampel yang diperlukan untuk berada di simpul daun
   
- **AdaBoost**

| **_Hyperparameter_** |       **Nilai**       |
|:--------------------:|:---------------------:|
|     n_estimators     |     [50, 100, 150]    |
|     learning_rate    | [0.01, 0.1, 0.5, 1.0] |

  - n_estimators = jumlah estimator (*weak learners*)
  - learning_rate = tingkat pembelajaran untuk menyesuaikan kontribusi setiap *weak learners*

Setelah dilakukan proses *tuning* menggunakan **GridSearch**, diperoleh nilai *hyperparameter* pada setiap model sebagai berikut:
- **KNNeighbors** (n_neighbors = 9, p = 2,  weights = uniform)
- **RandomForest** (max_depth = 10, min_samples_leaf =  2, min_samples_split = 10, n_estimators = 150)
- **AdaBoost** (learning_rate = 0.01, n_estimators = 150)

Hasil pelatihan model (metrik MSE akan dijelaskan pada bagian Evaluasi) menggunakan data latih ditunjukkan pada Tabel 3.

Tabel 3. Hasil Pelatihan Model pada Data Latih
|index|KNN|RandomForest|Boosting|
|---|---|---|---|
|train\_mse|20717419607\.23316|23395585501\.723976|24378081632\.521862|

## 6. Evaluation
Untuk menentukan kualitas dari model, digunakan metrik **MSE** atau *Mean Squared Error* sebagai metrik evaluasi model. MSE menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Semakin besar nilai MSE menunjukkan kualitas yang buruk pada model tersebut, sebaliknya nilai MSE yang semakin kecil menunjukkan kualitas model yang baik.

Selama proses pelatihan model menggunakan *hyperparameter* terbaiknya pada data latih dan data uji, diperoleh hasil MSE pada masing - masing model yang ditunjukkan pada Gambar 10 dan Tabel 4.

![mse_models](https://github.com/zulfianrahma/Predictive-Analytics-House-Pricing-Study-Case/assets/97383651/7b6a5ef1-6a28-4ac8-a644-ad7952d77085)

Gambar 10. Diagram Batang Hasil Pelatihan Model pada Data Latih dan Data Uji


Tabel 4. Hasil Pelatihan Model pada Data Latih dan Data Uji
|        **Model**        |  **train_mse**  | **test_mse**    |
|:-----------------------:|:---------------:|-----------------|
|       KNNeighbors       | 20717419.607233 | 60868329.802501 |
| Random Forest Regressor | 23395585.501724 | 58853949.443876 |
| AdaBoost Regressor      | 24378081.632522 | 55088400.939983 |

Berdasarkan hasil tersebut, **KNNeighbor** menunjukkan hasil terbaik di antara model lainnya pada tahap pelatihan dengan *hyperparameter* ** n_neighbors = 9, p = 1, dan weights = uniform **. Sedangkan untuk tahap pengujian, **AdaBoost** menunjukkan hasil terbaik di antara model lainnya dengan *hyperparameter* **learning_rate = 0.01, n_estimators = 100**

## 7. Daftar Pustaka
1.	S. Lahmiri, S. Bekiros, and C. Avdoulas, “A comparative assessment of machine learning methods for predicting housing prices using Bayesian optimization,” Decision Analytics Journal, vol. 6, p. 100166, Mar. 2023, doi: 10.1016/J.DAJOUR.2023.100166.
2.	C. Zhan, Y. Liu, Z. Wu, M. Zhao, and T. W. S. Chow, “A hybrid machine learning framework for forecasting house price,” Expert Syst Appl, vol. 233, pp. 957–4174, 2023, doi: 10.1016/j.eswa.2023.120981.
3.	A. B. Adetunji, O. N. Akande, F. A. Ajala, O. Oyewo, Y. F. Akande, and G. Oluwadara, “House Price Prediction using Random Forest Machine Learning Technique,” Procedia Comput Sci, vol. 199, pp. 806–813, Jan. 2022, doi: 10.1016/J.PROCS.2022.01.100.
4.	“K-Nearest Neighbor(KNN) Algorithm for Machine Learning - Javatpoint.” https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning (accessed Jul. 23, 2023).
5.	“K-Nearest Neighbors Algorithm - Intuitive Tutorials.” https://intuitivetutorial.com/2023/04/07/k-nearest-neighbors-algorithm/ (accessed Jul. 23, 2023).
6.	“Random Forest Regression. Random Forest Regression is a… | by Chaya | Level Up Coding.” https://levelup.gitconnected.com/random-forest-regression-209c0f354c84 (accessed Jul. 23, 2023).
7.	“sklearn.ensemble.AdaBoostRegressor — scikit-learn 1.3.0 documentation.” https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html (accessed Jul. 23, 2023).
 

**---Ini adalah bagian akhir laporan---**
