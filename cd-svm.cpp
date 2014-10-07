#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <Eigen/SparseCore>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/LU>
#include <algorithm>
int main(int argc, char* argv[])
{
  if (argc !=5)
    {
      std::cout<<"Invalid input"<<std::endl;
      return 1;
    }

  double C = atof(argv[1]);
  int num_threads = atoi(argv[2]);
  int num_rows;
  int num_cols;
  int entries_train;
  int entries_test;
  double optimum;
  std::string trainfile = argv[3];
  std::string testfile = argv[4];
  //std::cout<<trainfile<<std::endl;
  if (trainfile.compare("covtype.tr")==0) {
    entries_train = 500000;
    entries_test = 81012;
    num_cols = 54;
    num_rows = entries_train;
    optimum = 14672.5;
    //std::cout<<"Matched covtype.tr" <<std::endl;

  }
  else {
     
    entries_train = 677399;
    entries_test = 20242;
    num_cols = 47276;
    num_rows = entries_train;
    optimum = 4568.02;
  }
  std::ifstream input;

  Eigen::SparseMatrix <double> X(entries_train,num_cols);
  Eigen::SparseMatrix <double> Xtran(num_cols,entries_train);
  Eigen::SparseMatrix <double> Xt(entries_test,num_cols);
  Eigen::VectorXi Y(entries_train);
  Eigen::VectorXi Yt(entries_test);
  //std::cout<<"Entries train "<<entries_train<<std::endl;

  typedef Eigen::Triplet<double> T;
  std::vector<T> train_list,test_list,trans_list;
  //train_list.reserve(entries_train*num_cols);
  //test_list.reserve(entries_test*num_cols);
  //trans_list.reserve(entries_train*num_cols);
  int r,c,i;
  float val;
  std::string spaceDelimit = " ";
  std::string delimiter = ":";
  input.open(trainfile.c_str());
  if(!input)
    std::cout<<"Failed to open training file "<<trainfile<<std::endl;

  //std::cout<<"Opened training file "<<trainfile<<std::endl;
  r=0;
  double rel_error_array[20];
  double wall_time[20];
  while (!input.eof())
    {
      std::string line;
      getline(input,line);
      std::size_t pos = 0;
      i = 0;
      while ((pos = line.find(spaceDelimit)) != std::string::npos)
      {
	if(i==0)
	{
	  int label = atoi(line.substr(0,pos).c_str());
	  Y(r) = label;
	  i++;
	}
	else
	{
	  std::string temp = line.substr(0,pos);
	  std::size_t pos1 = temp.find(delimiter);

	  c = atoi(temp.substr(0,pos1).c_str());
	  val = atof(temp.substr(pos1+1).c_str());
	  // std::cout<<"temp "<<temp<<" pos1 "<<pos1<<" val "<<val<<std::endl;
	  train_list.push_back(T(r,c-1,val));
	  trans_list.push_back(T(c-1,r,val));
	}

	line.erase(0,pos+1);
      }
      r++;
    }

  //std::cout<<r<<std::endl;
  X.setFromTriplets(train_list.begin(),train_list.end());
  Xtran.setFromTriplets(trans_list.begin(),trans_list.end());
  //std::cout<<"Finished reading train file"<<std::endl;
  input.close();
  //std::cout<<X.nonZeros()<<std::endl;
  //std::cout<<Xtran.nonZeros()<<std::endl;
  input.open(testfile.c_str());
  std::vector<T>().swap(train_list);
  if(!input)
    std::cout<<"Failed to open test file"<<testfile<<std::endl;

  r = 0;
  //std::cout<<"Opened test file "<<testfile<<std::endl;
  
  
  while (!input.eof())
    {
      std::string line;
      getline(input,line);
      std::size_t pos = 0;
      i = 0;
      while ((pos = line.find(spaceDelimit)) != std::string::npos)
      {
	if(i==0)
	{
	  int label = atoi(line.substr(0,pos).c_str());
	  Yt(r) = label;
	  i++;
	}
	else
	{
	  std::string temp = line.substr(0,pos);
	  std::size_t pos1 = temp.find(delimiter);
	  c = atoi(temp.substr(0,pos1).c_str());
	  val = atof(temp.substr(pos1+1).c_str());
	  // std::cout<<val<<std::endl;
	  test_list.push_back(T(r,c-1,val));
	}

	line.erase(0,pos+1);
      }
      r++;
    }

  Xt.setFromTriplets(test_list.begin(),test_list.end());
  //std::cout<<"Finished reading test file"<<std::endl;
  //std::cout<<r<<std::endl;
  std::vector<T>().swap(test_list);
  //std::cout<<Xt.nonZeros()<<std::endl;
  int d = num_cols;
  int n = entries_train;
  Eigen::VectorXd alpha = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd weight = Eigen::VectorXd::Zero(d);

  std::vector <int> random_vector;
  for (int i = 0;i<n;i++)
    random_vector.push_back(i);

  //std::random_shuffle(random_vector.begin(),random_vector.end());
  
  //Eigen::MatrixXd Q(n,n);

      //Check test accuracy
    Eigen :: VectorXd predictions(entries_test);
    predictions = Xt*weight;
    double correct = 0;
    for (int l = 0; l<predictions.size();l++)
      {
	if(predictions[l]>0)
	  predictions[l] = 1;
	else
	  predictions[l] = -1;
	//std::cout<<l<<" "<<predictions[l]<<" "<<Yt[l]<<std::endl;
	if (Yt[l] == predictions[l])
	  correct++;
      }
    //std::cout<<correct<<" "<<entries_test<<std::endl;
    double accuracy = (correct/entries_test)*100;
    std::cout<<"Initial Prediction accuracy is "<<accuracy<<" % "<<std::endl;

  int max_iter = 20;
  double start = omp_get_wtime();
  for (int iter=0;iter<max_iter;iter++)
  {
    std::random_shuffle(random_vector.begin(),random_vector.end());
    std::cout<<"Iteration number "<<iter+1<<std::endl;
    
    //double start = omp_get_wtime();
    omp_set_num_threads(num_threads);
    //#pragma omp parallel for shared(weight) schedule(dynamic)
    for (int i = 0;i<n;i++)
    {
      int update_dim = random_vector[i];
      double Qii=1/(2*C);
      double G = 0.0;
      Eigen::VectorXd x_col = Eigen::VectorXd::Zero(d);
      for (Eigen::SparseMatrix<double>::InnerIterator it(Xtran,update_dim);it; ++it)
      {
	Qii += it.value()*it.value();
	//std::cout<<"val "<<it.value()<<" Qii "<<Qii<<std::endl;
	//update dim corresponds to col or data point it.row() is dim d
	G+=Y(update_dim)*weight(it.row())*it.value();
	x_col(it.row())=Y(update_dim)*it.value();
      }
      
      double old_alpha = alpha[update_dim];
      G = G - 1 + (old_alpha/(2*C));
      double PG;
      if(old_alpha==0.0)
	PG = std::min(G,0.0);

	  if(PG!=0)
	    {
	      //std::cout<<"Updating alpha at dim "<<update_dim<<" with value "<<old_alpha<<std::endl;
	      //std::cout<<" Qii "<<Qii<<" G"<<G-1<<std::endl;
	      alpha[update_dim] = std::max(old_alpha - (G/Qii),0.0);
	      //std::cout<<" New alpha "<<alpha[update_dim]<<std::endl;
	      // omp_lock_t writelock;
	      // omp_init_lock(&writelock);
	      // omp_set_lock(&writelock);
	      for (int wt = 0; wt<weight.size(); wt++)
		{
		  double temp = (alpha[update_dim]-old_alpha)*x_col[wt];
		  // #pragma omp atomic update
		  weight[wt]+=temp;
		}
	      //weight = weight + (alpha[update_dim] - old_alpha)*x_col;
	      //omp_unset_lock(&writelock);
	      // std::cout<<weight.sum()<<std::endl;
	    }
	  else
	    {
	    //std::cout<<"PG 0 didnt update alpha at dim "<<update_dim<<std::endl;
	    }
	  // }
     }
    wall_time[iter] = omp_get_wtime()-start;
    std::cout<<"Time for iteration "<<omp_get_wtime()-start<<std::endl;
    
    //Check sum(yi*alphai*xi)==0
    double error = 0.0;
    Eigen::VectorXd observed_weight(d);
    for (int k=0;k<X.outerSize();++k)
      { double observed_w = 0.0;
	for(Eigen::SparseMatrix<double>::InnerIterator it(X,k);it;++it)
	{
	  //std::cout<<"Check equal assumption "<<k<<" "<<it.col()<<std::endl;
	  observed_w += alpha[it.row()]*Y(it.row())*it.value();
	}
      observed_weight[k] = observed_w;
      error += pow(weight[k]-observed_weight[k],2);
      //std::cout<<"Observed error "<<error<<std::endl;
      }
    std::cout<<"Final error "<<sqrt(error)<<std::endl;
    
    //Check dual objective function value
    double dual = weight.transpose()*weight;
    dual = dual*0.5;
    double a_norm = alpha.transpose()*alpha;
    a_norm = a_norm/(4*C);
    dual = dual + a_norm;
    double a_sum = alpha.sum();
    dual = dual - a_sum;
    std::cout<<"Dual of objective function value "<<dual<<std::endl;
    //Check primal objective function value
    double primal = 0.5*weight.transpose()*weight;
    Eigen::VectorXd observations(entries_train);
    observations = X*weight;
    //Y = Y.cast<double>();
    //observations = observations.array()*Y.array();
    for (int l = 0; l<observations.size(); l++)
      {
	primal = primal + C*pow(std::max((1-observations[l])*Y[l],0.0),2);
      }
    
    std::cout <<"Primal objective function value "<<primal<<std::endl;
    //Check wall time for co-ordinate descent update
    
    double relative_error = 0.0;
    relative_error = log10(abs(primal-optimum))-log10(optimum);
    rel_error_array[iter] = relative_error;
    std::cout<<"Relative error "<<relative_error<<std::endl;

    //Check test accuracy
    Eigen :: VectorXd predictions(entries_test);
    predictions = Xt*weight;
    double correct = 0;
    for (int l = 0; l<predictions.size();l++)
      {
	if(predictions[l]>0)
	  predictions[l] = 1;
	else
	  predictions[l] = -1;
	//std::cout<<l<<" "<<predictions[l]<<" "<<Yt[l]<<std::endl;
	if (Yt[l] == predictions[l])
	  correct++;
      }
    //std::cout<<correct<<" "<<entries_train<<std::endl;
    double accuracy = (correct/entries_test)*100;
    std::cout<<"Prediction accuracy is "<<accuracy<<" % "<<std::endl;


    
    
  }
  //std::cout<<"Relative error "<<std::endl;
  for(int i = 0; i<20;i++)
    //std::cout<<wall_time[i]<<" "<<rel_error_array[i]<<std::endl;

  return 0;
}
