//LSSVM参数设定
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <cstdlib>
using namespace std;

//double sigma = 0.28;			//核参数 fatboy数据100-50
//double sigma = 0.49;			//核参数 housing
//double sigma = 0.50;			//dianjia核参数
//double gamma = 100;					//惩罚因子

//double sigma = 5.6;				//水量预测数据参数
//double gamma = 900; 

int scal_flag=0;				//归一化参数

int dim_flag = 8;				//输入数据的维数
int dim = 8;				//多维数据：数据维数
int MD = 20;				//一维数据：嵌入维数MD

double beta;						//b
vector<double> alpha;				//alpha

vector<vector<double> > train_x;		//训练数据
vector<double> train_y;
vector<vector<double> > test_x;			//测试数据
vector<double> test_y;

vector<double> pred;					//预测数据
vector<double> pred_best;				//最优预测数据

void Read_Data();		//读取数据
void Writ_Data();
void Scale();			//数据归一化,当scal_flag=0，归一化到[0,1]；=1 归一化到[-1,1]

double RBF(vector<double> xi, vector<double> xj, double sigma);	//RBF函数
void Kmatrix(vector<vector<double> > &KerlMatrix, double sigma, double gamma);	//构造核矩阵
void Gauss(vector<vector<double> > KerlMatrix);		//高斯消元法
void Predict(double sigma);					//预测函数
void Result_out();				//输出结果

//PSO参数设定
const int Pnum = 10;            //粒子数目
const int Dnum = 50;			//迭代总次数1000

const double SigLow  = 0.1;            //sigma搜索域范围
const double SigHigh = 10;
const double SigVmin = -10;			//sigma的速度范围
const double SigVmax = 10;

const double GamLow = 1;		        //gamma搜索域范围
const double GamHigh = 100;
const double GamVmin = -100;			//gamma的速度范围
const double GamVmax = 100;

const double c1 = 2;               //学习因子
const double c2 = 2;
const double Wmin=0.4;			//最小权值
const double Wmax=0.9;			//最大权值
double w = 0.9;					//初始权值
int wn = 1;

//PSO参数设定
double sigma[Pnum];			  //粒子集合
double gamma[Pnum];
double sigma_loc_best[Pnum];		//局部最优
double gamma_loc_best[Pnum];
double sigma_glo_best;				//全局最优
double gamma_glo_best;
double sigma_v[Pnum];			//更新速度
double gamma_v[Pnum];
double fit[Pnum];			//粒子当前的适应度
double loc_fit[Pnum];		//局部最优值
double gfit;				//全局最优适应值

//PSO优化相关函数
void Initial();								//初始化
double FitNess(double gamma, double sigma);	//计算适应度
void renew_par();							//更新粒子信息
void renew_w();						//更新权重

void WCJS();

int main()
{	
	clock_t begin = clock();
	
	Read_Data();
	Writ_Data();
	Scale();
	
	cout<<endl;
	cout<<"......训练样本数目 = "<<train_x.size()<<"......"<<endl;
	cout<<"......测试样本数目 = "<<test_x.size()<<"......"<<endl<<endl;

	cout<<"............开始训练............"<<endl<<endl;
	
	srand((unsigned)time(NULL));
	
	Initial();
	
	int num=0;
	while(num<Dnum)
	{
		//if(num%10 == 0)
		
		cout<<"num = "<<num+1<<"  ";
		cout<<"sigma_glo_best = "<<sigma_glo_best<<"  ";
		cout<<"gamma_glo_best = "<<gamma_glo_best<<endl;

		renew_par();
		renew_w();
		num++;
	}
	
	Result_out();
	
	clock_t end = clock();
	cout<<"程序运行时间： "<<(end-begin)/CLOCKS_PER_SEC<<"秒"<<endl<<endl;
	
	return 0;
}

void Read_Data()
{
	double data;	
	
	if( dim_flag != 1 )		//多维输入
	{
		vector<double> vec(dim);
		
		//fstream fin1("mpg_train.txt", fstream::in);
		//fstream fin1("house_train.txt", fstream::in);
		//fstream fin1("load_train.txt", fstream::in);
		
		fstream fin1("train.txt", fstream::in);
		for(int i=0; fin1>>data; i++)
		{
			train_y.push_back(data);

			for(int j=0; j<dim; j++)
			{
				fin1>>vec[j];
			}
			train_x.push_back(vec);
		}
		fin1.close();

		//fstream fin2("mpg_test.txt", fstream::in);
		//fstream fin2("house_test.txt", fstream::in);
		//fstream fin2("load_test.txt", fstream::in);
		fstream fin2("test.txt", fstream::in);
		for(int i=0; fin2>>data; i++)
		{
			test_y.push_back(data);

			for(int j=0; j<dim; j++)
			{
				fin2>>vec[j];
			}
			test_x.push_back(vec);
		}
		fin2.close();
		
		pred.resize( test_x.size() );
		pred_best.resize( test_x.size() );
	}
	
	if(dim_flag == 1)		//一维时间序列
	{
		dim = MD;
		fstream fin1("load_train.txt", fstream::in);
		
		vector<double> vec(dim);
		
		vector<double> train_series;
		vector<double> test_series;

		for(int i=0; fin1>>data; i++)
		{
			train_series.push_back(data);
		}

		for(int i=0; i<train_series.size()-dim; i++)
		{
			train_y.push_back(train_series[i+dim]);

			for(int j=0; j<dim; j++)
			{
				vec[j] = train_series[i+j];
			}
			train_x.push_back(vec);
		}
		fin1.close();

		fstream fin2("load_test.txt", fstream::in);
		for(int i=0; fin2>>data; i++)
		{
			test_series.push_back(data);
		}

		for(int i=0; i<test_series.size()-dim; i++)
		{
			test_y.push_back(test_series[i+dim]);

			for(int j=0; j<dim; j++)
			{
				vec[j] = test_series[i+j];
			}
			test_x.push_back(vec);
		}
		fin2.close();
		
		pred.resize( test_x.size() );
		pred_best.resize( test_x.size() );
	}
}

void Writ_Data()
{
	fstream cout("train_x.txt", fstream::out);
	for(int i=0; i<train_x.size(); i++)
	{
		for(int j=0; j<dim; j++)
			cout<<train_x[i][j]<<" ";
		cout<<endl;
	}
}

// =0归一化到[0,1]; =1 归一化到[-1,1]
void Scale()
{
	int i,j;
	
	while(scal_flag!=0 && scal_flag!=1 )
	{
		cout<<"scal_flag输入错误，只能输入0或者1,重新输入scal_flag:"<<endl;
		cin>>scal_flag;
		if(scal_flag==0 || scal_flag==1)
			break;
	}

	for(j=0; j!=dim; j++)
	{
		double max = train_x[0][j];
		double min = train_x[0][j];

		for(i=0; i!=train_x.size() ;i++)
		{
			if( train_x[i][j] > max )
				max = train_x[i][j];
			if( train_x[i][j] < min )
				min = train_x[i][j];
		}

		for(i=0; i!=test_x.size() ;i++)
		{
			if( test_x[i][j] > max )
				max = test_x[i][j];
			if( test_x[i][j] < min )
				min = test_x[i][j];
		}
		//cout<<"max="<<max<<"  min="<<min<<endl;
		if(max == min)
		{
			for(i=0; i!=train_x.size() ;i++)
				train_x[i][j] = max;
			for(i=0; i!=test_x.size() ;i++)
				test_x[i][j] = max;
			continue;
		}

		if(scal_flag == 0)//归一化到[0,1]
		{
			for(i=0; i!=train_x.size() ;i++)
			{
				train_x[i][j] = (train_x[i][j]-min)/(max-min);
			}
			for(i=0; i!=test_x.size() ;i++)
			{
				test_x[i][j] = (test_x[i][j]-min)/(max-min);
			}
		}
		if( scal_flag == 1 )//归一化到[-1,1]
		{
			for(i=0; i!=train_x.size() ;i++)
			{
				train_x[i][j] = (train_x[i][j]-max-min)/(max-min);
			}
		
			for(i=0; i!=test_x.size() ;i++)
			{
				test_x[i][j] = (2*test_x[i][j]-max-min)/(max-min);
			}
		}
	}
	
	//归一化输出
	fstream cout("scale.txt", fstream::out);

	cout<<"train_x:"<<endl;
	for(i=0; i!=train_x.size() ;i++)
	{
		for(j=0; j!=train_x[0].size(); j++)
		{
			cout<<train_x[i][j]<<" ";
		}
		cout<<endl;
	}
	
	cout<<"test_x:"<<endl;
	for(i=0; i!=test_x.size() ;i++)
	{
		for(j=0; j!=test_x[0].size(); j++)
		{
			cout<<test_x[i][j]<<" ";
		}
		cout<<endl;
	}
}

//RBF核函数
double RBF(vector<double> xi, vector<double> xj, double sigma)
{
	double sum=0;
	for(int i=0; i<xi.size(); i++)
	{
		sum += (xi[i]-xj[i])*(xi[i]-xj[i]);
	}
	return exp( (-1)*sum/2.0/sigma/sigma );
}

//构造核矩阵
void Kmatrix(vector<vector<double> > &KerlMatrix, double sigma, double gamma)
{
	fstream Kerl_out("Kerl.txt");
	
	int i,j,k;
	int num = train_x.size();
	
	KerlMatrix[0][0] = 0;
	for(k=1; k<num+1; k++)
	{
		KerlMatrix[0][k] = 1;
		KerlMatrix[k][0] = 1;
	}

	for(i=1; i<num+1; i++)
	{
		for(j=1; j<num+1; j++)
		{
			if(i==j)
				KerlMatrix[i][j] = RBF(train_x[i-1],train_x[j-1], sigma)-1.0/gamma;
			else
				KerlMatrix[i][j] = RBF(train_x[i-1],train_x[j-1], sigma);
		}
	}

	KerlMatrix[0][num+1] = 0;
	for(i=1; i<num+1; i++)
	{
		KerlMatrix[i][num+1]=train_y[i-1];
	}
	
	//*
	for(i=0; i<KerlMatrix.size(); i++)
	{
		for(j=0; j<KerlMatrix.size(); j++)
			Kerl_out<<KerlMatrix[i][j]<<" ";
		Kerl_out<<endl;
	}
	Kerl_out<<endl;//*/
}

//高斯消元
void Gauss(vector<vector<double> > KerlMatrix)
{
	fstream fout("kout.txt", fstream::out);

	int i,j,k, ii, jj, ir ;
	int N = KerlMatrix.size();
	
	//每列相加替代第一行
	double sum;
	for(j=0; j<N+1; j++)
	{
		sum=0;
		for(i=0; i<N; i++)
			sum += KerlMatrix[i][j];
		KerlMatrix[0][j]=sum;
	}
			
	//化成上三角
	double temp;
	for(i=0; i<N; i++)
	{
		//如果KerlMatrix[i][i]=0, 交换
		if(KerlMatrix[i][i]==0)
		{
			for(ir=i+1; ir<N; ir++)
			{
				if(KerlMatrix[ir][i]!=0)
				{
					for(jj=0; jj<N+1; jj++)
					{
						temp = KerlMatrix[ir][jj];
						KerlMatrix[ir][jj] = KerlMatrix[i][jj];
						KerlMatrix[i][jj] = KerlMatrix[ir][jj];;
					}
					break;
				}
			}
		}

		//化成上三角
		temp = KerlMatrix[i][i];
		for(j=i; j<N+1; j++)
		{
			KerlMatrix[i][j] = KerlMatrix[i][j]/temp;
		}
	
		for(j=i+1; j<N; j++)
		{
			temp = KerlMatrix[j][i];
			for(k=i+1; k<N+1; k++)
				KerlMatrix[j][k] -= temp*KerlMatrix[i][k];
			KerlMatrix[j][i] = 0;
		}
	}


	//求解
	for(j=N-1; j>0; j--)
	{
		for(i=j-1; i>=0; i--)
		{
			temp = KerlMatrix[i][j];
			KerlMatrix[i][N] -= temp * KerlMatrix[j][N];
			KerlMatrix[i][j] -= temp * KerlMatrix[j][j];	
		}
	}
	
	/*输出矩阵
	for(ii=0; ii<KerlMatrix.size(); ii++)
	{
		for(jj=0; jj<KerlMatrix.size(); jj++)
			cout<<KerlMatrix[ii][jj]<<" ";
		cout<<endl;
	}
	cout<<endl;
	//*/
	
	//系数
	alpha.resize(N-1);
	for(i=1; i<N; i++)
	{
		alpha[i-1] = KerlMatrix[i][N] ;
		//cout<<alpha[i-1]<<endl;
	}
	beta = KerlMatrix[0][N];
	//cout<<beta<<endl;
}

void Predict(double sigma)
{
	fstream fout("rbf.txt", fstream::out);

	for(int i=0; i!=test_x.size(); i++)
	{	
		pred[i] = 0 ;
		for(int j=0; j != alpha.size(); j++)
		{
			//fout<<"RBF = "<<RBF(test_x[i], train_x[j], sigma)<<endl;
			pred[i] += alpha[j]*RBF(test_x[i], train_x[j], sigma);
		}
		//fout<<endl;
		pred[i] += beta;
	}
}

void Result_out()
{
	fstream out("result.txt", fstream::out);

	cout<<endl<<"预测结果输出到resule.txt文件"<<endl<<endl;

	cout<<"gfit = "<<gfit<<endl;
	cout<<"sigma_glo_best = "<<sigma_glo_best<<endl;
	cout<<"gamma_glo_best = "<<gamma_glo_best<<endl;

	double ape;
	double sum_ape=0;

	out<<left<<setw(15)<<"真实值"<<setw(15)<<"预测值"<<setw(15)<<" 百分比误差（APE）"<<endl;

	for(int i=0; i!=test_y.size(); i++)
	{	
		ape = abs(test_y[i]-pred_best[i])/abs(test_y[i])*100;
		sum_ape += ape;

		out<<left<<setw(15)<<test_y[i]<<setw(15)<<pred_best[i]<<setw(15)<<ape<<endl;
	}
	cout<<"Sum_APE = "<<sum_ape<<"   APE = "<<sum_ape/test_y.size()<<endl;
	out<<"Sum_APE = "<<sum_ape<<"   APE = "<<sum_ape/test_y.size()<<endl;
}


void Initial()
{
	int i,j;
	for(i=0; i<Pnum; i++)
	{
		sigma[i] = SigLow + (SigHigh - SigLow)*1.0*rand()/RAND_MAX;		//初始化群体[SigLow,SigHigh]
		gamma[i] = GamLow + (GamHigh - GamLow)*1.0*rand()/RAND_MAX;		//初始化群体[GamLow,GamHigh]
		
		sigma_loc_best[i] = sigma[i];					//sigma的局部最优
		gamma_loc_best[i] = gamma[i];					//gamma的局部最优
	
		sigma_v[i] = -SigVmax + 2* SigVmax*1.0*rand()/RAND_MAX;    //随机生成a的速度[-SigVmax,SigVmax]
		gamma_v[i] = -GamVmax + 2* GamVmax*1.0*rand()/RAND_MAX;    //随机生成c的速度[-GamVmax,GamVmax]
	}

	//找出全局最优适应值
	j=0;
	for(i=0; i<Pnum; i++)
	{
		fit[i] = FitNess(sigma[i],gamma[i]);		//每个粒子的适应度
		loc_fit[i] = fit[i];					//局部最优值
		
		if(i==0)
		{
			gfit = loc_fit[0];
			for(int k=0; k<pred.size(); k++)
			{
				pred_best[k] = pred[k];
			}
		}

		if( i>0 && loc_fit[i] < gfit )
		{
			gfit = loc_fit[i];
			j = i;	
			for(int k=0; k<pred.size(); k++)
			{
				pred_best[k] = pred[k];
			}
		}
	}
	
	//全局最优向量
	sigma_glo_best = sigma_loc_best[j];
	gamma_glo_best = gamma_loc_best[j];
}

double FitNess(double sigma, double gamma)
{
	int N = train_x.size()+1;
	vector<vector<double> > KerlMatrix(N,vector<double>(N+1));	//构造核矩阵
	
	Kmatrix( KerlMatrix, sigma, gamma);
	Gauss( KerlMatrix );
	Predict( sigma );
	
	double error=0;
	for(int i=0; i<test_y.size(); i++)
	{
		error += abs( pred[i]-test_y[i] )/abs( test_y[i] )*100;
	}
	return error/test_y.size();
}

void renew_w()
{
	w = Wmax-wn*(Wmax-Wmin)/Dnum;//惯性权重的改变
	wn++;
}

void renew_par()
{
	int i,j;
	
	//更新粒子位置
	for(i=0; i<Pnum; i++)
	{
		sigma[i] += sigma_v[i];
		gamma[i] += gamma_v[i];
		
		if(sigma[i] > SigHigh)
		{
			sigma[i] = SigHigh;
		}
		if(sigma[i] < SigLow)
		{
			sigma[i] = SigLow;
		}
		if(gamma[i] > GamHigh)
		{
			gamma[i] = GamHigh;
		}
		if(gamma[i] < GamLow)
		{
			gamma[i] = GamLow;
		}
	}

	
	for(i=0; i<Pnum; i++)            
	{
		//计算每个粒子的适应度
		fit[i] = FitNess(sigma[i],gamma[i]);
      
		//更新个体局部最优值
		if(fit[i] < loc_fit[i])
		{
			loc_fit[i] = fit[i];
			sigma_loc_best[i] = sigma[i];
			gamma_loc_best[i] = gamma[i];
		}
		
		//更新全局适应值
		if(loc_fit[i] < gfit)
		{
			gfit = loc_fit[i];

			for(int k=0; k<pred.size(); k++)
			{
				pred_best[k] = pred[k];
			}
			//更新全局最优向量
			
			sigma_glo_best= sigma_loc_best[i];
			gamma_glo_best = gamma_loc_best[i];
		}
	}

	cout<<"gfit = "<<gfit<<endl;
	

	//更新个体速度
	for(i=0; i<Pnum; i++)    
	{
		sigma_v[i] = w*sigma_v[i]+
			c1*1.0*rand()/RAND_MAX*(sigma_loc_best[i] - sigma[i])+
			c2*1.0*rand()/RAND_MAX*(sigma_glo_best - sigma[i]);
		gamma_v[i] = w*gamma_v[i]+
			c1*1.0*rand()/RAND_MAX*(gamma_loc_best[i] - gamma[i])+
			c2*1.0*rand()/RAND_MAX*(gamma_glo_best - gamma[i]);
		
		if(sigma_v[i] > SigVmax)
		{
			sigma_v[i] = SigVmax;
		}
		if(sigma_v[i] < SigVmin)
		{
			sigma_v[i] = SigVmin;
		}
		
		if(gamma_v[i] > GamVmax)
		{
			gamma_v[i] = GamVmax;
		}
		if(gamma_v[i] < GamVmin)
		{
			gamma_v[i] =  GamVmin;
		}
	}
}
