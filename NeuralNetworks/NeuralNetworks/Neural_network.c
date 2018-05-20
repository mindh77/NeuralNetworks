#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Windows.h>

#define NLayer 3  // 총 뉴론 네트워크 층 수
#define MLayerSize 1000 // 각 층 당 최대 뉴론의 개수
#define m0 80 // 층 1의 뉴론의 수
#define m1 120 // 층 2의 뉴론의 수
#define m2 10 // 최종 층의 뉴론의 수
#define N 784+1 // 하나의 training의 입력의 수(dummy input 포함)
#define N_tr_examples 60000 // 총 training의 개수 (한 epoch)
#define N_te_examples 10000 // 총 test의 개수
double c= 0.05; // Weight Update 시 사용되는 임의의 실수 값

double s[NLayer][MLayerSize] = { 0.0, };
double f[NLayer][MLayerSize] = { 0.0, };
double delta[NLayer][MLayerSize] = { 0.0, };
double W[NLayer][MLayerSize][MLayerSize] = { 0.0, };


int M[NLayer] = { m0,m1,m2 };
//모든 층의 뉴론의 수를 가진 배열

int input[N];
//training data의 한 입력벡터를 읽어서 이 변수에 채움
int D[m2];
//training data의 기대하는 출력 값으로 이 변수를 채움
//만약 6이라면 ( 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 )
int TrainingData[N_tr_examples][N - 1];
int d_tr[N_tr_examples][m2]= { 0, };
//train.txt로 부터 읽어들인 data들을 저장해 놓는다

int TestData[N_te_examples][N - 1];
int d_te[N_te_examples][m2] = { 0, };
//test.txt로 부터 읽어들인 data들을 저장해 놓는다

//평균 에러
double sum_sq_error = 0.0f;
double avg_sq_error = 0.0f; 
double avg_sq_error_before = 0.0f;

//Weight Vector값을 초기화
void initiate_weight_vector();

//trainingdata와 testdata를 가져와 TrainingData, d_tr과 TestData,d_te를 채워넣어줌
void get_training_data();
void get_test_data();

//하나의 training data에 대해서 다음 3개의 함수를 차례대로 실행한다.
void forward_compute(); 
void backward_compute();
void weight_update();
//한 epoch 후에 평균 에러 avg_sq_error를 계산
void avg_sq_error_compute();

void show_result();
void gotoxy(int x, int y);
//훈련이 종료된 후에 testdata를 이용하여 시스템의 성능을 측정.
void test();

int main() {
	int set_cnt = 0; //60000개의 training set를 카운트
	int data_cnt = 0; //784개의 데이터 카운트
	int index_for_D = 0;
	int err_set_cnt = 0; //60000개의 training set를 카운트
	int err_data_cnt = 0; //784개의 데이터 카운트
	int err_index_for_D = 0;
	int i = 0;
	int j = 0;
	int k = 0;
	int l = 0;
	int m = 0;
	initiate_weight_vector();
	get_training_data();
	get_test_data();
	while (1) { //6만개의 데이터 forward 시행 후 오차율 계산.
		for (err_set_cnt = 0; err_set_cnt < N_tr_examples; err_set_cnt++) {
			for (err_data_cnt = 0; err_data_cnt < 784; err_data_cnt++) {
				input[k] = TrainingData[err_set_cnt][err_data_cnt];
				k++;
			}
			input[k] = 1;
			k = 0;
			l = 0;
			for (err_index_for_D = 0; err_index_for_D < 10; err_index_for_D++) {
				D[l] = d_tr[err_set_cnt][err_index_for_D];
				l++;
			}
			forward_compute();
			for (m = 0; m<M[NLayer - 1]; m++)
				sum_sq_error += (D[m] - f[NLayer - 1][m])* (D[m] - f[NLayer - 1][m]);
			
		}
		for (set_cnt = 0; set_cnt < N_tr_examples; set_cnt++) {
			for (data_cnt=0; data_cnt < 784; data_cnt++) {
				input[i] = TrainingData[set_cnt][data_cnt];
				i++;
			}
			input[i] = 1;
			i = 0;
			j = 0;
			for (index_for_D=0; index_for_D < 10; index_for_D++) {
				D[j] = d_tr[set_cnt][index_for_D];
				j++;
			}
			forward_compute();
			backward_compute();
			weight_update();
		}
		avg_sq_error_compute();
		
		show_result();
		test();

		if (avg_sq_error < 0.02) {
			break;
		}
		avg_sq_error_before = avg_sq_error;
		if (c > 0.002) {
			c -= 0.001;
		}
	}

	return 0;
}

void initiate_weight_vector() {
	int i;
	int j;
	int k;
	int pre_Layer;
	//neuron j 로 들어 오는 입력 신호의 개수 (dummy input 포함)
	printf("Weight Vector 초기화 시작.\n");

	srand(time(NULL));
	for (i = 0; i < NLayer; i++) {
		for (j = 0; j < M[i]; j++) {
			//첫 번째 층일 경우, 들어오는 입력의 수를 N, 즉 입력의 수로 결정
			if (i == 0) {
				pre_Layer = N;
			}
			//나머지 층 들은, 전 층의 총 뉴론의 수 +1(dummy input)으로 결정
			else {
				pre_Layer = M[i - 1] + 1;
			}
			for (k = 0; k < pre_Layer; k++) {
				W[i][j][k] = (double)(rand()) / (double)RAND_MAX-0.5;
			}
		}
	}
	printf("Weight Vector 초기화 완료.\n");
}
void get_training_data() {
	int i=0; //d_tr[60000][m2]의 60000을 카운트 해 줄 변수
	int row; //28개의 행
	int col; //28개의 열
	int cnt; //784개까지 카운트 되고 다시 0으로
	int d=0; //train data의 각 train의 첫 줄에 있는 정답 값을 가져와 저장
	
	FILE* fp;
	printf("트레이닝 데이터를 가지고 옵니다.\n");
	fp = fopen("train.txt", "r");
	for(i=0;i<N_tr_examples;i++) {
		cnt = 0;
		fscanf(fp, "%d\n",&d);
		d_tr[i][d] = 1;
		for (row = 0; row < 28; row++) {
			for (col = 0; col < 27; col++) {
				fscanf(fp, "%d ", &TrainingData[i][cnt]);
				cnt++;
			}
			fscanf(fp, "%d\n", &TrainingData[i][cnt]);
			cnt++;
		}

	}
	printf("트레이닝 데이터를 모두 가져왔습니다.\n");
	fclose(fp);
}
void get_test_data() {
	int i = 0; //d_tr[60000][m2]의 60000을 카운트 해 줄 변수
	int row; //28개의 행
	int col; //28개의 열
	int cnt; //784개까지 카운트 되고 다시 0으로
	int d = 0; //train data의 각 train의 첫 줄에 있는 정답 값을 가져와 저장

	FILE* fp;
	printf("테스트 데이터를 가지고 옵니다.\n");
	fp = fopen("test.txt", "r");
	for (i = 0; i<N_te_examples; i++) {
		cnt = 0;
		fscanf(fp, "%d\n", &d);
		d_te[i][d] = 1;
		for (row = 0; row < 28; row++) {
			for (col = 0; col < 27; col++) {
				fscanf(fp, "%d ", &TestData[i][cnt]);
				cnt++;
			}
			fscanf(fp, "%d\n", &TestData[i][cnt]);
			cnt++;
		}

	}
	printf("테스트 데이터를 모두 가져왔습니다.\n");
	fclose(fp);
}
void forward_compute() {
	int i;
	int j;
	int L; //Layer
	// 층 0 의 각 뉴론 i 마다에 대하여 s, f 계산함.
	for (i = 0; i<M[0]; i++) {  
		s[0][i] = 0.0; // 초기화!!
		for (j = 0; j < N; j++) {    // 이 뉴론으로 들어오는 j 번째 신호에 대하여,
			s[0][i] += input[j] * W[0][i][j];   // 주의: s 초기화한 후 해야 함.
		}
		f[0][i] = 1.0 / (1.0 + exp(-s[0][i]));
		f[0][m0] = 1.0; // 더미입력을 넣어 놓음.
	}
	for (L = 1; L< NLayer; L++) {   //  층 L 에 대해서,
		for (i = 0; i< M[L]; i++) {  //  L 층의 뉴론 i 에 대하여 s, f 계산함.
			s[L][i] = 0.0; // 초기화!!
			for (j = 0; j < (M[L - 1] + 1); j++) {   // j : 이전 층에서 뉴론 i 로 들어오는 신호의 번호. +1 은 더미입력을 포함하기 위함.
				s[L][i] += f[L - 1][j] * W[L][i][j];
			}
			// f 의 계산        
			f[L][i] = 1.0 / (1.0 + exp(-s[L][i]));
		}
		f[L][M[L]] = 1.0;  // 더미입력 넣음.
	}
}
void backward_compute() {
	int k;
	int i;
	int j;
	double tsum;
	int L;
	k = NLayer - 1;  // 최종층 번호
	for (i = 0; i<M[k]; i++)  // 최종층의  각 뉴론 i 에 대하여,
		delta[k][i] = (D[i] - f[k][i])*f[k][i] * (1 - f[k][i]);
	for (L = (NLayer - 2); L >= 0; L--) {  // L : 층번호. 
		for (i = 0; i < M[L]; i++) { // 층 L의 뉴론 i 에 대하여 delta 계산하자.
			tsum = 0.0;
			for (j = 0; j < M[L + 1]; j++) { // 이전 층의 뉴론 j 의 delta 이용.
				tsum += delta[L + 1][j] * W[L + 1][j][i]; // 이용되는 connection: 뉴론 j로의 i-th 입력
			}
			delta[L][i] = f[L][i] * (1 - f[L][i]) * tsum;
		}
	}
}
void weight_update() {
	int i;
	int j;
	int L;
	// L=0 인 경우 즉 층 0 의 뉴론에 대한 처리
	for (i = 0; i<M[0]; i++)   // 뉴론 i 의 모든 입력 신호에 대한 가중치 갱신
		for (j = 0; j < N; j++)  // 해당 뉴론으로의  입력 신호 j 마다에 대하여 (dummy 신호 포함)
			W[0][i][j] += c * delta[0][i] * input[j];

	// L > 0 인 층들의 뉴론의 가중치들에 대한 처리.
	for (L = 1; L< NLayer; L++)    // L: 층번호
		for (i = 0; i<M[L]; i++)        // i: 뉴론 번호
			for (j = 0; j < (M[L - 1] + 1); j++)   // j: 뉴론 i 으로 들어오는 입력신호 연결선 번호.
												   // 앞줄에서 +1 을 한 이유는 더미입력선을 위함.
				W[L][i][j] += c * delta[L][i] * f[L - 1][j];
}
void sum_sq_error_compute() {
	int t;
	int i;
	for (i = 0; i<M[NLayer - 1]; i++)
		sum_sq_error += (D[i] - f[NLayer - 1][i])* (D[i] - f[NLayer - 1][i]);
}
void avg_sq_error_compute() {
	avg_sq_error = sum_sq_error / (double)(N_tr_examples * M[NLayer - 1]);
	sum_sq_error = 0.0f;
}
void show_result() {
	static int cnt=1;
	printf("%d# epoch 수행 결과 평균 에러 : ", cnt);
	printf("%.3f%%\n",avg_sq_error*100);
	cnt++;
}
void test() {
	int set_cnt = 0;
	int data_cnt = 0;
	int i=0;
	int j=0;
	int index_for_D=0;
	int m=0;
	int max_value_index = 0;
	int ans_cnt = 0;
	double avg;

	for (set_cnt = 0; set_cnt < N_te_examples; set_cnt++) {
		for (data_cnt = 0; data_cnt < 784; data_cnt++) {
			input[i] = TestData[set_cnt][data_cnt];
			i++;
		}
		input[i] = 1;
		i = 0;
		j = 0;
		for (index_for_D = 0; index_for_D < 10; index_for_D++) {
			D[j] = d_te[set_cnt][index_for_D];
			j++;
		}
		forward_compute();
		for (m = 1; m < M[NLayer - 1]; m++) {
			if (f[NLayer - 1][m] >= f[NLayer - 1][max_value_index]) {
				max_value_index = m;
			}
		}
		if (D[max_value_index] == 1) {
			ans_cnt++;
		}
	}
	avg = (double)ans_cnt / (double)N_te_examples;
	printf("Test 데이터에서 정답을 맞춘 비율 : %.3f\n\n", avg*100);
}
