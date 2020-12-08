//Read in a graph. Compute persistence/robustness and critical points.

//2d persistence with killer idea, topbottom dimension

// #define VERBOSE 1

#include <exception>
#include <math.h>
#include <queue>
#include <stack>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <assert.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <numeric>
using namespace std;

// python thingy
#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

#include <pybind11/stl.h>

#define LOG_FILE "pers_MRF_2D_log.txt"

#ifdef MATLAB
extern void _main();

// Function declarations.
// -----------------------------------------------------------------
double getMatlabScalar (const mxArray* ptr) {

  // Make sure the input argument is a scalar in double-precision.
  if (!mxIsDouble(ptr) || mxGetNumberOfElements(ptr) != 1)
    mexErrMsgTxt("The input argument must be a double-precision scalar");

  return *mxGetPr(ptr);
}
double& createMatlabScalar (mxArray*& ptr) { 
  ptr = mxCreateDoubleMatrix(1,1,mxREAL);
  return *mxGetPr(ptr);
}

void myMessage (const string msg,bool showtime){
//	mexWarnMsgTxt(msg.c_str());
//	mexPrintf("%s\n",msg.c_str());
	
	 time_t now;
	 time(&now);

	fstream filestr;
	filestr.open (LOG_FILE, fstream::in | fstream::out | fstream::ate);
	
	if (showtime){
	  filestr << ctime(&now) << "-------" << msg.c_str() << endl;
	}else{
	  filestr << msg.c_str() << endl;
	}
	filestr.close();

}
#else
void myMessage (const string msg,bool showtime){
	time_t now;
	 time(&now);
// CCCC uncomment if you want output information
#ifdef VERBOSE
	if (showtime){
	  cout << ctime(&now) << "-------" << msg.c_str() << endl;
	}else{
	  cout<<msg.c_str()<<endl;
	}
#endif
}
#endif

#define  OUTPUT_MSG(MSG)  {stringstream * ss=new stringstream(stringstream::in | stringstream::out); *ss << MSG << ends; myMessage(ss->str(),true); delete ss; }

#define  OUTPUT_NOTIME_MSG(MSG)  {stringstream * ss=new stringstream(stringstream::in | stringstream::out); *ss << MSG << ends; myMessage(ss->str(),false); delete ss; }

// Function definitions.
// -----------------------------------------------------------------

void myassert(int ln, bool a){
	if (!a){
		OUTPUT_MSG("Line " << ln << " : ASSERTION FAILED!!!!!!!");
	}
        if(!a){
            cout << "ASSERTION FAILED at line " << ln << endl << flush;
//            do{
//                int x = 10;
//            }while(true);
        }
	assert(a);
	return;
}

class myDoubleMatrix {
	public:
		int nrow;
		int ncol;
		vector< vector< double > > data;
  myDoubleMatrix(int m,int n,double v=0.0) { 
	  nrow=m;
	  ncol=n;
	  int i,j;
	  for (i=0;i<nrow;i++)
		  data.push_back(vector< double >(ncol,v));
	  return;
  } 
  double get(int r, int c) { 
	 	myassert( __LINE__ , (0<=r)&&(r<nrow));
	 	myassert( __LINE__ , (0<=c)&&(c<ncol));
	     return data[r][c];
  }
  void put(int r, int c, double v) { 
	 	myassert( __LINE__ , (0<=r)&&(r<nrow));
	 	myassert( __LINE__ , (0<=c)&&(c<ncol));
	        data[r][c]=v;
  }
  void input1D(double *ptr){
	  int i,j;
	  for (i=0;i<nrow;i++)
		  for (j=0;j<ncol;j++)
			  data[i][j]=ptr[j*nrow+i];
  }
  void output1D(double *ptr){
	  int i,j;
	  for (i=0;i<nrow;i++)
		  for (j=0;j<ncol;j++)
			  ptr[j*nrow+i]=data[i][j];
  }
  static void output2DDoubleArray(double * ptr, vector<vector<double> > * M){
	int i,j;
        myassert( __LINE__ ,  M != NULL );
        myassert( __LINE__ ,  ! M->empty() );
        int size_each_row = (* M)[0].size();
        int tmp_nrow = M->size();
	for (i=0;i<M->size();i++){
            myassert( __LINE__ , size_each_row == (* M)[i].size() );
	    for (j=0;j<size_each_row;j++)
		    ptr[j*tmp_nrow+i]=(* M)[i][j];
        }
  }

  static vector<double> input1DArray(double * ptr, int len){
       int i;
       vector<double> ret(len);
       for (i = 0; i < len; ++i)
	    ret[i]=ptr[i];
       return ret;
    }
  static void output1DIntArray(double * ptr, vector<int> arr){
       int i;
       for (i = 0; i < arr.size(); ++i)
	    ptr[i] = (double) arr[i];
       return;
    }

};  

class Vertex{
	public:
	int vid;
	double fvalue;
	Vertex(int i = -1, double f = 0.0) : 
	vid(i),fvalue(f){}
};

class Edge{
	public:
	int v1_order;
	int v2_order;
        double fvalue;
	Edge(int v1o = -1, int v2o = -1, double f = 0.0) : 
	v1_order(min(v1o,v2o)),v2_order(max(v1o,v2o)),fvalue(f){
            if( v1_order == v2_order )
                cout << "Wrong Edge ( " << v1_order << " , " << v2_order << " ) " << endl;
            myassert( __LINE__ , v1_order != v2_order);
        }
};

class Triangle{
	public:
	int v1_order;
	int v2_order;
	int v3_order;
	int e1_order;
	int e2_order;
	int e3_order;
        double fvalue;
	Triangle(int v1o,int v2o,int v3o,int e1o,int e2o,int e3o,double f):
	fvalue(f){
		vector< int >tmpvec (3,0);
		tmpvec[0]=v1o;
		tmpvec[1]=v2o;
		tmpvec[2]=v3o;
		sort(tmpvec.begin(),tmpvec.end());
		v1_order=tmpvec[0];
		v2_order=tmpvec[1];
		v3_order=tmpvec[2];

		vector< int >tmpEdgevec (3,0);
		tmpEdgevec[0]=e1o;
		tmpEdgevec[1]=e2o;
		tmpEdgevec[2]=e3o;
		sort(tmpEdgevec.begin(),tmpEdgevec.end());
		e1_order=tmpEdgevec[0];
		e2_order=tmpEdgevec[1];
		e3_order=tmpEdgevec[2];
	}
};

bool vCompVal(Vertex a, Vertex b){ return a.fvalue < b.fvalue; }

bool eComp(Edge a, Edge b){ 
	if(a.v2_order!=b.v2_order)
		return a.v2_order<b.v2_order;
	return a.v1_order<b.v1_order;
}

bool trigComp(Triangle a, Triangle b){ 
	if(a.e3_order!=b.e3_order)
		return a.e3_order<b.e3_order;
	if(a.e2_order!=b.e2_order)
		return a.e2_order<b.e2_order;
	return a.e1_order<b.e1_order;
}

#define BIG_INT 0xFFFFFFF
template <class Container>
struct Counter : public std::iterator <std::output_iterator_tag,
                         void, void, void, void>
{ 
	size_t &cnt;

    Counter(size_t &x) : cnt(x) {}	
 
	template<typename t>
    Counter& operator=(t)
	{        
        return *this;
    }
    
    Counter& operator* () 
	{
        return *this;
    }
    
    Counter& operator++(int) 
	{
		++cnt;
        return *this;
    }    

	Counter& operator++() 
	{
		++cnt;
        return *this;
    }    
};

// We avoid excessive allocations by calculating the size of the resulting list.
// Then we resize the result and populate it with the actual values.
vector< int > list_sym_diff(vector< int > &sa, vector< int > &sb){
	//assume inputs are both sorted increasingly	
	size_t count = 0;
	Counter< vector< int > > counter(count);
	set_symmetric_difference(sa.begin(), sa.end(), sb.begin(), sb.end(), counter);
	vector< int > out;	
	
	out.reserve(count);
	set_symmetric_difference(sa.begin(), sa.end(), sb.begin(), sb.end(), back_inserter(out));	

	return out;
}


//-----------------------------------------------------
//vertex-edge pair and persistence
//edge-trig pair and persistence
//-----------------------------------------------------
class VertEdgePair{
  public:
  int vborder;
  int edorder;
  int vbidx;
  int vdidx;
  double robustness;
  double birth;
  double death;
  
  //initialize coordinates using the input vertices and persistence
  VertEdgePair(int vbo, int edo, int vbi, int vdi, double rob, double b, double d) : 
  vborder(vbo),edorder(edo),
  vbidx(vbi),vdidx(vdi),
  robustness(rob),birth(b),death(d){}

  bool operator<(const VertEdgePair &rhs) const{
    return (this->robustness >= rhs.robustness);
  }
};

class EdgeTrigPair{
  public:
  int eborder, tdorder;
  int vbidx, vdidx; 
  double robustness;
  double birth;
  double death;
  
  //initialize coordinates using the input vertices and persistence
  EdgeTrigPair( int ebo, int tdo, int vbi, int vdi, double rob,double b,double d) : 
  eborder(ebo),tdorder(tdo),vbidx(vbi),vdidx(vdi),
  robustness(rob),birth(b),death(d){}

  bool operator<(const EdgeTrigPair &rhs) const{
    return (this->robustness >= rhs.robustness);
  }
};

//-----------------------------------------------------
//compute 2D persistence
// m,n: size of the two dimensions
// pers_thd: threshold of persistence (only bigger persistence would be recorded
// rob_thd: threshold of robustness
// levelset_val: the image value of the levelset (0 in image segmentation)
// persistenceM: persistence flow, +pers to creator and -pers to destroyer
// robustnessM: robustness flow, +pers to creator or -pers to destroyer, depending on which is closer to the levelset_val 
// veList: vertex-edge pair, together with corresponding persistence
// etrigList: edge-triangle pair, together with corresponding persistence
//
// assume the global variable phi is already available (which stores the height function)
//-----------------------------------------------------
#define MAX_PERS_PTS 1000	//maximum of numbers of persistence pts


	void buildBoundary2D(vector<vector < int > > * boundary_2D, vector<Triangle> * trigList, int trigNum, int nedge){
                myassert( __LINE__ , (trigNum > 0) && (nedge > 0));
		int i, j, idx;
		for (i=0; i<trigNum; i++){
			(* boundary_2D)[i].push_back( (* trigList)[i].e1_order );
			(* boundary_2D)[i].push_back( (* trigList)[i].e2_order );
			(* boundary_2D)[i].push_back( (* trigList)[i].e3_order );
		}
	}
					
	void buildBoundary1D(vector<vector < int > > * boundary_1D, vector<Edge> * eList, int nedge, int nvert){
                myassert( __LINE__ , (nvert > 0) && (nedge > 0));
		int i, j, idx;
		for (i=0; i<nedge; i++){
			(* boundary_1D)[i].push_back( (* eList)[i].v1_order );
			(* boundary_1D)[i].push_back( (* eList)[i].v2_order );
		}
	}
	
void remove_redundant(vector<int> & vec){
            sort(vec.begin(), vec.end());
            vector<int>::iterator tmp_iter = unique(vec.begin(), vec.end());
            vec.resize( distance(vec.begin(), tmp_iter) );
}

vector<int> calcPers(vector<vector<int> > knnG, vector<double> vert_f, int nv, 
	      const double rob_thd, const double levelset_val,
		bool skip1d, vector<vector<double> > * persList,
                vector<bool> incorr_pred_as_curr_label, 
                vector<int> * incorr_comp_vert_list = NULL,
                vector<int> true_labels = vector<int>(),
                vector<int> pred_labels = vector<int>(),
                vector<vector<int> > * to_return = NULL
                ){

	OUTPUT_MSG("Begin computing persistence");

        int nvert = nv;                 // nvert is the number of training data, 
                                        // build persistence on
        int ntestv = vert_f.size() - nvert;

	OUTPUT_MSG("nvert = " << nvert);
	OUTPUT_MSG("ntestv = " << ntestv);

	//constructing vList
	int i,j,k;

	vector< Vertex > * vList=new vector< Vertex >(nvert);
	vector< Edge > * eList=new vector< Edge >();
        
        // create and sort vert list
	for (i=0;i<nvert;i++)
	    (* vList)[i] = Vertex(i,vert_f[i]);
	sort(vList->begin(), vList->end(), vCompVal);
        int global_min_vidx = (* vList)[0].vid;

        //create a map from vid to vorder
        map<int, int> mapVid2Vorder;
        int vid, vid2, vorder, vorder2;
        for(i = 0;i < nvert; ++i){
            vid = (* vList)[i].vid;
            myassert( __LINE__ ,  mapVid2Vorder.find(vid) == mapVid2Vorder.end() );
            mapVid2Vorder[vid] = i;
        }

        // translate edge_list into eList
        double tmpd, tmpd2;
	map<pair<int, int>, int> tmpMapVo2E;
	pair<int, int> tmpipair;
        
        for (i=0;i<nvert;i++){
            vid = i;
            for (j = 0; j < knnG[i].size(); ++j){
                vid2 = knnG[i][j];
                myassert( __LINE__ ,  mapVid2Vorder.find(vid) != mapVid2Vorder.end() );
                myassert( __LINE__ ,  mapVid2Vorder.find(vid2) != mapVid2Vorder.end() );
                // skip self-loop (this could happen with hnsw approximation KNN)
                // myassert( __LINE__ ,  vid != vid2 );
                if(vid == vid2) continue;   

                vorder = mapVid2Vorder[vid];
                vorder2 = mapVid2Vorder[vid2];
                tmpd = (* vList)[vorder].fvalue;
                tmpd2 = (* vList)[vorder2].fvalue;

	        tmpipair = pair<int, int>(min(vorder,vorder2), max(vorder,vorder2)); 
                if(tmpMapVo2E.find(tmpipair) == tmpMapVo2E.end()){
                    // if the pair has not been seen, insert them as a new edge
                    eList->push_back(Edge(vorder, vorder2, max(tmpd, tmpd2)));
                    tmpMapVo2E[tmpipair] = eList->size() - 1;
                }
            }
        }
        // sort eList
	sort(eList->begin(),eList->end(),eComp);
        int nedge = eList->size();
        
	OUTPUT_MSG("nedge = " <<nedge);

        // construct neighbor list for each vid (not vorder)
        vector<vector<int> > neighbor_list(nvert, vector<int>());
        vector<int>::iterator tmp_iter;
        for (i=0;i<nvert;i++){
            vid = i;
            for (j = 0; j < knnG[i].size(); ++j){
                vid2 = knnG[i][j];
                myassert( __LINE__ ,  (vid2 >= 0) && (vid2 < nvert) );
                
                // skip self-loop (this could happen with hnsw approximation KNN)
                // myassert( __LINE__ ,  vid != vid2 );
                if(vid == vid2) continue;   
                neighbor_list[vid].push_back(vid2);
                neighbor_list[vid2].push_back(vid);
            }
        }
        // clean up duplicates
        for (i = 0; i < nvert; ++i){
            sort(neighbor_list[i].begin(), neighbor_list[i].end());
            tmp_iter = unique(neighbor_list[i].begin(), neighbor_list[i].end());
            neighbor_list[i].resize( distance(neighbor_list[i].begin(), tmp_iter) );
        }
    
	OUTPUT_MSG(__LINE__)

        // if test data are given, identify their neighbors from training data
        vector<vector<int> > nb_list_test2train(ntestv, vector<int>()); 
        if(ntestv > 0){
            // special neighbor list, for test vertices only
            for (i=nvert;i<nvert+ntestv;i++){
                vid = i;
                for (j = 0; j < knnG[i].size(); ++j){
                    vid2 = knnG[i][j];
                    myassert( __LINE__ ,  (vid2 >= 0) && (vid2 < nvert) );
                    
                    myassert( __LINE__ ,  vid != vid2 );
                    nb_list_test2train[vid-nvert].push_back(vid2);
                }
            }
            for (i = 0; i < ntestv; ++i){
                sort(nb_list_test2train[i].begin(), nb_list_test2train[i].end());
                tmp_iter = unique(nb_list_test2train[i].begin(), nb_list_test2train[i].end());
                nb_list_test2train[i].resize( distance(nb_list_test2train[i].begin(), tmp_iter) );
            }
        } 
        
	OUTPUT_MSG(__LINE__)

//         // copy eList to the final_edge_list (return out to matlab for debugging)
//         final_edge_list->resize(nedge);
//         for (i = 0; i < nedge; ++i){
//             (* final_edge_list)[i].resize(2);
//             vorder = (* eList)[i].v1_order;
//             vorder2 = (* eList)[i].v2_order;
//             (* final_edge_list)[i][0] = (* vList)[vorder].vid;
//             (* final_edge_list)[i][1] = (* vList)[vorder2].vid;
//         }

	OUTPUT_MSG("--vList and eList constructed and sorted");

        // create trig_list only if need 1D homology
	vector< Triangle > * trigList = NULL;
	vector< Triangle > * trigListBrutal = NULL;
        int e1o, e2o, e3o, v1o, v2o, v3o;
        if(! skip1d){  
            //mapping vert order to edge order
	    map<pair<int, int>, int> emapVo2Eo;
	    pair<int, int> tmpipair;
	    for(i = 0; i < eList->size(); ++i){
	        tmpipair = pair<int, int>((* eList)[i].v1_order, (* eList)[i].v2_order); 
	        emapVo2Eo[tmpipair] = i;
	    }

            trigListBrutal = new vector< Triangle >();
            // create triglist bruteforce
            for(i = 0; i < nvert; ++i)
                for(j = i+1; j < nvert; ++j)
                    for(k = j+1; k < nvert; ++k){
                       if( ( emapVo2Eo.find(pair<int,int>(i,j)) != emapVo2Eo.end() ) &&
                           ( emapVo2Eo.find(pair<int,int>(j,k)) != emapVo2Eo.end() ) &&
                           ( emapVo2Eo.find(pair<int,int>(i,k)) != emapVo2Eo.end() ) ){
                            e1o = emapVo2Eo[pair<int,int>(i,j)];
                            e2o = emapVo2Eo[pair<int,int>(j,k)];
                            e3o = emapVo2Eo[pair<int,int>(i,k)];
                            myassert( __LINE__ ,  (*vList)[i].fvalue <= (*vList)[j].fvalue);
                            myassert( __LINE__ ,  (*vList)[j].fvalue <= (*vList)[k].fvalue);
                            trigListBrutal->push_back(Triangle(i,j,k,e1o,e2o,e3o,(*vList)[k].fvalue));
                       }
                    }
            sort(trigListBrutal->begin(), trigListBrutal->end(), trigComp);
            
            // next implementation, more efficient, assume edges are sparse
            // build a vo2eo maps, only map to each edge once (by the smaller vo)
            vector<vector<int> > adj_list(nvert, vector<int>() );
            for(i = 0; i < nedge; ++i){
                // cout << "Edge " << i << " = ( " << (* eList)[i].v1_order << " , " <<(* eList)[i].v2_order << " ) " << endl;
                int v1o = (* eList)[i].v1_order;
                adj_list[v1o].push_back(i);
            }
            trigList = new vector< Triangle >();
            for(i = 0; i < nvert; ++i)
                for(j = 0; j < adj_list[i].size(); ++j)
                    for(k = j+1; k < adj_list[i].size(); ++k){
                        e1o = adj_list[i][j];
                        e2o = adj_list[i][k];
                        v1o = i;
                        myassert( __LINE__ ,  (* eList)[e1o].v1_order == v1o );
                        myassert( __LINE__ ,  (* eList)[e2o].v1_order == v1o );
                        v2o = (* eList)[e1o].v2_order;
                        v3o = (* eList)[e2o].v2_order;
                        myassert( __LINE__ ,  v2o < v3o );
                        
                        if( emapVo2Eo.find(pair<int,int>(v2o, v3o)) != emapVo2Eo.end() ){
                            e3o = emapVo2Eo[pair<int,int>(v2o, v3o)];
                            if(e3o <= e2o){
                                cout << e1o << " " << e2o << " " << e3o << endl;
                                cout << v1o << " " << v2o << " " << v3o << endl;
                                cout << (* eList)[e1o].v1_order << " " <<  (* eList)[e1o].v2_order << endl; 
                                cout << (* eList)[e2o].v1_order << " " <<  (* eList)[e2o].v2_order << endl; 
                                cout << (* eList)[e3o].v1_order << " " <<  (* eList)[e3o].v2_order << endl; 
                            }
                            myassert( __LINE__ , e3o > e1o );
                            myassert( __LINE__ , e3o > e2o );
                            tmpd = (* vList)[v3o].fvalue; 
                            trigList->push_back(Triangle(v1o,v2o,v3o,e1o,e2o,e3o,tmpd));
                        }
                    }

            sort(trigList->begin(), trigList->end(), trigComp);
// 
//             for(i = 0; i < 10 ; ++i){
//                 cout << (* vList)[(* trigList)[i].v1_order].vid << " "
//                      << (* vList)[(* trigList)[i].v2_order].vid << " "
//                      << (* vList)[(* trigList)[i].v3_order].vid << " "
//                      << (* vList)[(* trigListBrutal)[i].v1_order].vid << " "
//                      << (* vList)[(* trigListBrutal)[i].v2_order].vid << " "
//                      << (* vList)[(* trigListBrutal)[i].v3_order].vid << endl;
//             }
// 
//            cout << trigList->size() << " " << trigListBrutal->size() << endl;

            // sanity check, the two triglists should be identical
            myassert( __LINE__ , trigList->size() == trigListBrutal->size());
            for(i = 0; i < trigList->size(); ++i){
                Triangle trig1 = (* trigList)[i];
                Triangle trig2 = (* trigListBrutal)[i];
                myassert( __LINE__ , trig1.v1_order == trig2.v1_order);
                myassert( __LINE__ , trig1.v2_order == trig2.v2_order);
                myassert( __LINE__ , trig1.v3_order == trig2.v3_order);
                myassert( __LINE__ , trig1.e1_order == trig2.e1_order);
                myassert( __LINE__ , trig1.e2_order == trig2.e2_order);
                myassert( __LINE__ , trig1.e3_order == trig2.e3_order);
                myassert( __LINE__ , fabs(trig1.fvalue - trig2.fvalue)<=0.0001);
            }
        } // end of if(! skip1d) 
        
        int trigNum = (trigList == NULL) ? 0 : trigList->size();

	//construct and reduce 2d boundary matrix
	vector< vector< int > > * boundary_2D= NULL;
        vector< int > * low_2D_e2t= NULL;
        
	multiset< EdgeTrigPair > etQueue;	// robustness pairs

	int num_e_creator=0;
	int num_t_destroyer=0;
	int num_v_creator=0;// number of vertices creating non-essential class
	int num_e_destroyer=0;// number of edge destroyer (non-essential)

	//output edge-trig pairs whose persistence is bigger than pers_thd
	int vBirth,vDeath,vBirthIdx,vDeathIdx;
	int tmp_int;
	double tmp_pers,tmp_rob,tmp_double,tmp_death,tmp_birth;
	
	int low;

	list<int>::iterator myiter;
	int tmpe12,tmpe23,tmpe13;
	map<int, int>::iterator tmpit;

        if(!skip1d && trigNum > 0){  
            boundary_2D = new vector< vector< int > >(trigNum, vector< int >()); //first index is col index, each col init to empty
	    buildBoundary2D(boundary_2D, trigList, trigNum, nedge);
	    low_2D_e2t = new vector< int >(nedge,-1);
	    for (i=0;i<trigNum;i++){

		//reduce column i
		low = * (* boundary_2D)[i].rbegin();

		while ( ( ! (* boundary_2D)[i].empty() ) && ( (* low_2D_e2t)[low]!=-1 ) ){
			(* boundary_2D)[i]=list_sym_diff((* boundary_2D)[i],(* boundary_2D)[(* low_2D_e2t)[low]]);

			if(! (* boundary_2D)[i].empty()){
				low = * (* boundary_2D)[i].rbegin();
			}
		}
		if (! (* boundary_2D)[i].empty()){
			myassert( __LINE__ , low>=0);
			myassert( __LINE__ , (* low_2D_e2t)[low]==-1);
			(* low_2D_e2t)[low]=i;
			num_t_destroyer++;
			num_e_creator++;

			//record pair
			Edge edgeCreator=(* eList)[low];
			Triangle trigDestroyer=(* trigList)[i];
			vBirth= edgeCreator.v2_order;
			vDeath= trigDestroyer.v3_order;

                        vBirthIdx = (* vList)[vBirth].vid;
                        vDeathIdx = (* vList)[vDeath].vid;

			tmp_death= (* vList)[vDeath].fvalue;
			tmp_birth= (* vList)[vBirth].fvalue;
			tmp_rob=min(fabs(tmp_birth-levelset_val),fabs(tmp_death-levelset_val));
			if( (tmp_birth<levelset_val) && (tmp_death>levelset_val) && (tmp_rob>rob_thd) ){
				etQueue.insert(EdgeTrigPair(low,i,vBirthIdx,vDeathIdx,tmp_rob, tmp_birth, tmp_death));	
			};

		}

		if (i % 100000 == 0)
		  OUTPUT_MSG( "reducing boundary 2D: i=" << i <<", trig number=" << trigNum );
            }
            
	    delete boundary_2D;
	    OUTPUT_MSG( "boundary_2D all reduced" );
	} // end of if(!skip1d){  
	
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//construct and reduce 1d boundary matrix

	vector< int > * low_1D_v2e= new vector< int >(nvert,-1);
	// for each creator vertex, store the edge it is paired to
	
	vector< vector< int > > * boundary_1D=new vector< vector< int > >(nedge, vector< int >()); //first index is col index, each col init to empty
	buildBoundary1D(boundary_1D, eList, nedge, nvert);

	multiset< VertEdgePair > veQueue;	// robustness pairs

	for (i=0;i<nedge;i++){

            if(!skip1d && trigNum > 0){
                myassert( __LINE__ , low_2D_e2t != NULL);
		if ( (* low_2D_e2t)[i] >= 0 ){ 
			(*boundary_1D)[i].clear();
		    continue;
		}else{
		    myassert( __LINE__ , (* low_2D_e2t)[i] == -1);
		    myassert( __LINE__ , (*boundary_1D)[i].size()==2);
		};
            } 

		//reduce column i
		low = * (* boundary_1D)[i].rbegin();

		while ( ( ! (* boundary_1D)[i].empty() ) && ( (* low_1D_v2e)[low]!=-1 ) ){
			(* boundary_1D)[i]=list_sym_diff((* boundary_1D)[i],(* boundary_1D)[(* low_1D_v2e)[low]]);
			if(! (* boundary_1D)[i].empty()){
				low = * (* boundary_1D)[i].rbegin();
			}
		}
		if (! (* boundary_1D)[i].empty()){
			myassert( __LINE__ , low>=0);
			myassert( __LINE__ , (* low_1D_v2e)[low]==-1);
			(* low_1D_v2e)[low]=i;
			num_e_destroyer++;
			num_v_creator++;

			myassert( __LINE__ , (* boundary_1D)[i].size()==2);

// // Reduce the high entry of the remaining column, unnecessary
// 			int high =  * (* boundary_1D)[i].begin();
// 			//reduce high
// 			while (  (* low_1D_v2e)[high]!=-1 ){
// 				int edge_high=(*low_1D_v2e)[high];
// 				(* boundary_1D)[i]=list_sym_diff((* boundary_1D)[i],(* boundary_1D)[edge_high]);
// 				myassert( __LINE__ , (* boundary_1D)[i].size()==2);
// 				high = * (* boundary_1D)[i].begin();
// 			}

			//record pair
			vBirth= low;	//creator vertex
			Edge eDestroyer = (* eList)[i];
			vDeath=eDestroyer.v2_order;

                        vBirthIdx = (* vList)[vBirth].vid;
                        vDeathIdx = (* vList)[vDeath].vid;

			tmp_death= (* vList)[vDeath].fvalue;
			tmp_birth= (* vList)[vBirth].fvalue;

			tmp_rob=min(fabs(tmp_birth-levelset_val),fabs(tmp_death-levelset_val));
			if( (tmp_birth<levelset_val) && (tmp_death>levelset_val) && (tmp_rob>rob_thd) ){
				veQueue.insert(VertEdgePair(low,i,vBirthIdx,vDeathIdx,tmp_rob, tmp_birth, tmp_death));	
				//the component could be killed by either merge or remove
			};

		}else{
			// myassert( __LINE__ , false);
                        num_e_creator ++;
		}

		if (i % 100000 == 0)
		  OUTPUT_MSG( "reducing boundary 1D: i=" << i <<", edge number=" << nedge );
	}   // end of for (i=0;i<nedge;i++){

        // record paired v list, indexed by actual id, not vorder
        // a paired v creates a persistence pair
        // all component creator and the global min are not paired
        vector<bool> is_v_paired(nvert, false);
        for(i = 0; i < nvert; ++i){
            if( (* low_1D_v2e)[i] < 0 )
                continue;
            myassert( __LINE__ , (* low_1D_v2e)[i] < nedge );
            vid = (* vList)[i].vid;     // from vorder to vid
            myassert( __LINE__ , (vid >= 0) && (vid < nvert) );
            is_v_paired[vid] = true;
        }

	OUTPUT_MSG( "boundary 1D all reduced" );

        int ncomp = nvert - num_v_creator; 
        int nloop = nedge - num_e_destroyer; 
        myassert( __LINE__ , nloop == num_e_creator );
        int nvoid = trigNum - num_t_destroyer;
	myassert( __LINE__ , ncomp >= 1);
        myassert( __LINE__ , nloop >= 0);
        myassert( __LINE__ , nvoid >= 0);
        if(!skip1d)
	    myassert( __LINE__ , num_e_destroyer==num_v_creator);

        if (low_2D_e2t != NULL)
	    delete low_2D_e2t;
	delete low_1D_v2e;
	delete vList;
	delete eList;
        if (trigList != NULL)
	    delete trigList;
        if (trigListBrutal != NULL)
	    delete trigListBrutal;
	delete boundary_1D;
	

        // collect all un_paired vertex, create robust pairs
        double global_max = * max_element(vert_f.begin(), vert_f.begin() + nvert);
        double global_min = * min_element(vert_f.begin(), vert_f.begin() + nvert);
        OUTPUT_MSG( "Global max = " << global_max );
        OUTPUT_MSG( "Global min = " << global_min );
        for(i = 0; i < nvert; ++i){
            if(! is_v_paired[i]){
		tmp_birth = vert_f[i];
                if(tmp_birth < levelset_val){
		    tmp_rob = fabs(tmp_birth-levelset_val);
                    veQueue.insert(VertEdgePair(-1,-1,i,-1, tmp_rob, tmp_birth, 1000000.0) ); // death time be just bigger than global max, easier to print
                }
            }
        }

	for(multiset< VertEdgePair >::iterator myveiter=veQueue.begin(); myveiter!=veQueue.end(); myveiter++){	
            persList->push_back(vector<double>(6));
            int curr_idx = persList->size() - 1;
            (* persList)[curr_idx][0] = 0;
            (* persList)[curr_idx][1] = myveiter->vbidx;
            (* persList)[curr_idx][2] = myveiter->vdidx;
            (* persList)[curr_idx][3] = myveiter->birth;
            (* persList)[curr_idx][4] = myveiter->death;
            (* persList)[curr_idx][5] = myveiter->robustness;
        }

        if(!skip1d){
	    for(multiset< EdgeTrigPair >::iterator myetiter=etQueue.begin(); myetiter!=etQueue.end(); myetiter++){	
                persList->push_back(vector<double>(6));
                int curr_idx = persList->size() - 1;
                (* persList)[curr_idx][0] = 1;
                (* persList)[curr_idx][1] = myetiter->vbidx;
                (* persList)[curr_idx][2] = myetiter->vdidx;
                (* persList)[curr_idx][3] = myetiter->birth;
                (* persList)[curr_idx][4] = myetiter->death;
                (* persList)[curr_idx][5] = myetiter->robustness;
            }
        }

	OUTPUT_MSG(__LINE__)

        // label all connected components
        int comp_id, tmpv;
        int unassigned_cid = -10;
        vector<int> comp_map(nvert+ntestv, unassigned_cid);
        queue<int> myvqueue;
        for(j = 0; j < persList->size(); ++j){
            if((* persList)[j][0] != 0) break;
            comp_id = j;
            vid = (* persList)[j][1];  // birth vertex id
            myvqueue.push(vid);
            myassert( __LINE__ ,  vert_f[vid] < levelset_val);
            myassert( __LINE__ ,  comp_map[vid] == unassigned_cid);
            comp_map[vid] = comp_id;
            while(! myvqueue.empty()){
               tmpv = myvqueue.front();
               myvqueue.pop();
               myassert( __LINE__ ,  tmpv < nvert);
               myassert( __LINE__ ,  vert_f[tmpv] < levelset_val);
               for(i = 0; i < neighbor_list[tmpv].size(); ++i){
                   vid2 = neighbor_list[tmpv][i];
                   if(vert_f[vid2] < levelset_val){
                       if(comp_map[vid2] == unassigned_cid){
                            comp_map[vid2] = comp_id;
                            myvqueue.push(vid2);
                       }else{
                            myassert( __LINE__ ,  comp_map[vid2] == comp_id );
                       }
                   }
               }
            }
        }
    
        // sanity check
        for(i = 0; i < nvert; ++i){
            if(vert_f[i] < levelset_val)
                myassert( __LINE__ ,  comp_map[i] >= 0 );
            else
                myassert( __LINE__ ,  comp_map[i] == unassigned_cid );
        }

	OUTPUT_MSG(__LINE__)

        // color comp_id on test data (majority vote)
        map<int, int> tmp_map;
        map<int, int>::iterator tmp_map_iter;
        int cid, best_comp_ct;
        if(ntestv > 0){
            for(vid = nvert; vid < nvert+ntestv; ++vid){
                tmp_map.clear();
                for(i = 0; i < nb_list_test2train[vid-nvert].size(); ++i){
                    vid2 = nb_list_test2train[vid-nvert][i];
                    myassert( __LINE__ ,  (vid2 >= 0) && (vid2 < nvert) );
                    cid = comp_map[vid2];
                    if( cid != unassigned_cid){
                        if(tmp_map.find(cid) == tmp_map.end())
                            tmp_map[cid] = 1;
                        else
                            tmp_map[cid] = tmp_map[cid] + 1;
                    }
                }
                best_comp_ct = 0;
                for(tmp_map_iter = tmp_map.begin(); tmp_map_iter != tmp_map.end(); ++tmp_map_iter){
                    if(tmp_map_iter->second > best_comp_ct){
                        best_comp_ct = tmp_map_iter->second;
                        comp_map[vid] = tmp_map_iter->first;
                    }
                }
            }
        }

	OUTPUT_MSG(__LINE__)

        // get statistics: 
        // for each component: size, 
        // how many incorrectly predicted as curr comp (thus need to remove)
        vector<int> comp_counter(persList->size(), 0), comp_test_counter(persList->size(), 0); 
        vector<int> comp_incorr_train(persList->size(), 0), comp_incorr_test(persList->size(), 0);

        bool is_incorr;
        myassert( __LINE__ , comp_map.size() == nvert + ntestv );
        myassert( __LINE__ , incorr_pred_as_curr_label.size() == nvert + ntestv );
        for(i = 0; i < nvert; ++ i){
            cid = comp_map[i];
            is_incorr = incorr_pred_as_curr_label[i];
            if( cid >= 0 ){
                comp_counter[cid] ++;
                if(is_incorr)
                    comp_incorr_train[cid] ++;
            }else{
                myassert( __LINE__ , cid == unassigned_cid );
            }
        }
        for(i = nvert; i < nvert+ntestv; ++ i){
            cid = comp_map[i];
            is_incorr = incorr_pred_as_curr_label[i];
            if( cid >= 0 ){
                comp_test_counter[cid] ++;
                if(is_incorr)
                    comp_incorr_test[cid] ++;
            }else{
                myassert( __LINE__ , cid == unassigned_cid );
            }
        }
        
	OUTPUT_MSG(__LINE__)

        vector<int> comp_incorr_test_aggressive(persList->size(), 0);
        vector<int> tmp_comp_list;
        vector<int>::iterator myiter2;
        for(i = 0; i < ntestv; ++i){
            // if(i % 100 == 0)
            //     cout << "At " << i << endl;
            tmp_comp_list.clear();
            for(j = 0; j < nb_list_test2train[i].size(); ++j){
                vid2 = nb_list_test2train[i][j];
                myassert( __LINE__ , (vid2 >= 0) && (vid2 < nvert) );
                if(vert_f[vid2] < levelset_val){
                    cid = comp_map[vid2];
                    myassert( __LINE__ , cid != unassigned_cid );
                    tmp_comp_list.push_back(cid);
                }
            }

            // remove duplicates
            sort(tmp_comp_list.begin(), tmp_comp_list.end() );
            myiter2 = unique(tmp_comp_list.begin(), tmp_comp_list.end());
            tmp_comp_list.resize( distance(tmp_comp_list.begin(), myiter2) );

            for(j = 0; j < tmp_comp_list.size(); ++ j){
                cid = tmp_comp_list[j];
                myassert( __LINE__ , (cid >= 0)&&(cid < persList->size()) );
                comp_incorr_test_aggressive[cid] ++;
            }
        }

        // add results to persList 
        for(i = 0; i < persList->size(); ++i){
            myassert( __LINE__ , (* persList)[i].size() == 6);
            (* persList)[i].push_back(comp_counter[i]);
            (* persList)[i].push_back(comp_incorr_train[i]);
            (* persList)[i].push_back(comp_test_counter[i]);
            (* persList)[i].push_back(comp_incorr_test[i]);
            (* persList)[i].push_back(comp_incorr_test_aggressive[i]);
        }

        vector<int> ret; 
        ret.push_back(veQueue.size());
        ret.push_back(etQueue.size());
        ret.push_back(ncomp);
        ret.push_back(nloop);
        ret.push_back(nvoid);
        ret.resize(11);

	OUTPUT_MSG(__LINE__)

        if(persList->empty()){
            cout << "WARNING: NO COMPONETS TO FIX" << endl << flush;
            
            to_return->resize(6);
            (* to_return)[0].resize(3);
            return  ret;
        }
            
        OUTPUT_MSG("Num comp = " << comp_counter.size())
        OUTPUT_MSG("Num comp = " << persList->size())
        int max_comp_id = max_element(comp_counter.begin(), comp_counter.end()) - comp_counter.begin();
        // if need to return list of incorrect vertices within small componets
        if(incorr_comp_vert_list != NULL){
            // find the largest component
            OUTPUT_MSG( "########## Max comp id = " << max_comp_id << "; size = " << comp_counter[max_comp_id] )

            int num_small_comp_vert = 0;
            for(i = 0; i < nvert; ++i){
                if( (comp_map[i] != unassigned_cid) && (comp_map[i] != max_comp_id) ){
                    num_small_comp_vert ++;
                    if( incorr_pred_as_curr_label[i] )
                        incorr_comp_vert_list->push_back(i);
                }
            }

        }

        // collect all small components
	OUTPUT_MSG(__LINE__)

        
        // if need to return additional information
        if(to_return != NULL){
            comp_counter[max_comp_id] = 0; // to find the second largest
            int size_second_largest = * max_element(comp_counter.begin(), comp_counter.end());
            int total_size = accumulate(comp_counter.begin(),comp_counter.end(),0);
            to_return->resize(2);
            (* to_return)[0].push_back(total_size);  // counter of all small comps
            (* to_return)[0].push_back(size_second_largest); // size of second largest comp
            // (* to_return)[0].push_back(persList->size()); // size of second largest comp
            (* to_return)[0].push_back(persList->size()-1); // size of second largest comp


            // list of small comp vertices
            for(i = 0; i < nvert; ++i)
                if( (comp_map[i] != unassigned_cid) && (comp_map[i] != max_comp_id) )
                    (* to_return)[1].push_back(i);
             
           
	    OUTPUT_MSG(__LINE__)

            vector<bool> small_comp_bool(nvert, false);
            vector<bool> small_birth_bool(nvert, false);
            vector<bool> small_crit_bool(nvert, false);
            vector<bool> small_robCP_bool(nvert, false);
            //bool include_neighbors = true;
            bool include_neighbors = false;
            for(i = 0; i < nvert; ++i)
                if( (comp_map[i] != unassigned_cid) && (comp_map[i] != max_comp_id) )
                    small_comp_bool[i] = true;

            OUTPUT_MSG( "persList size: " <<  persList->size() )
            for(i = 0; i < persList->size(); ++i){
                if( i == max_comp_id ) continue;
                if( (* persList)[i][0] > 0 ) continue; // skip 1D

	        // OUTPUT_MSG("i = "<< i << " perslist[i] size " << (* persList)[i].size())

                int bidx = (* persList)[i][1];
                int didx = (* persList)[i][2];
                double btime = (* persList)[i][3];
                double dtime = (* persList)[i][4];
                bool kill_birth = (didx < 0) || (fabs(btime - levelset_val) <= fabs(dtime - levelset_val));
                // cout << bidx << " - " << didx << " - " << btime << " - " << dtime << " - " << kill_birth << endl << flush;
                small_birth_bool[bidx] = true;
                small_crit_bool[bidx] = true;
                if(didx >= 0){
                    myassert( __LINE__ , didx < nvert );
                    small_crit_bool[didx] = true;
                }
                if(include_neighbors){
                    for(j = 0; j < neighbor_list[bidx].size(); ++j){
                        vid2 = neighbor_list[bidx][j];
                        small_birth_bool[vid2] = true;
                        small_crit_bool[vid2] = true;
                    }
                    if(didx >= 0){
                        myassert( __LINE__ , didx < nvert );
                        for(j = 0; j < neighbor_list[didx].size(); ++j){
                            vid2 = neighbor_list[didx][j];
                            small_crit_bool[vid2] = true;
                        }
                    }
                }
                if(kill_birth){
                    myassert( __LINE__, (bidx >= 0) && (bidx < nvert) );
                    myassert( __LINE__, (!small_robCP_bool[bidx]) );
                    small_robCP_bool[bidx] = true;
                    if(include_neighbors){
                        for(j = 0; j < neighbor_list[bidx].size(); ++j){
                            vid2 = neighbor_list[bidx][j];
                            small_robCP_bool[vid2] = true;
                        }
                    }
                }else{
                    myassert( __LINE__, (didx >= 0) && (didx < nvert) );
                    small_robCP_bool[didx] = true;
                    if(include_neighbors){
                        for(j = 0; j < neighbor_list[didx].size(); ++j){
                            vid2 = neighbor_list[didx][j];
                            small_robCP_bool[vid2] = true;
                        }
                    }
                }
            }

	    OUTPUT_MSG(__LINE__)

        // -------- collect data to fix ---------
        vector<int> small_comp_vlist;
        for(i = 0; i < nvert; ++i)
            if((comp_map[i] != unassigned_cid) && (comp_map[i] != max_comp_id))
                small_comp_vlist.push_back(i);
        vector<int> small_birth_vlist;
        for(i = 0; i < nvert; ++i)
            if(small_birth_bool[i])
                small_birth_vlist.push_back(i);
        vector<int> small_crit_vlist;
        for(i = 0; i < nvert; ++i)
            if(small_crit_bool[i])
                small_crit_vlist.push_back(i);
        vector<int> small_robCP_vlist;
        for(i = 0; i < nvert; ++i)
            if(small_robCP_bool[i])
                small_robCP_vlist.push_back(i);
 
        myassert( __LINE__, small_robCP_vlist.size() <= small_birth_vlist.size() );
        myassert( __LINE__, small_robCP_vlist.size() <= small_crit_vlist.size() );
        to_return->push_back(small_comp_vlist);
        to_return->push_back(small_birth_vlist);
        to_return->push_back(small_crit_vlist);
        to_return->push_back(small_robCP_vlist);

            // list of wrong prediction vertices in testing
            myassert( __LINE__ , ! true_labels.empty());
            myassert( __LINE__ , ! pred_labels.empty());
            vector<bool> wrong_pred_test(ntestv, false);
            vector<int> wrong_pred_test_vlist;
            for(i = 0; i < ntestv; ++i)
                if(true_labels[nvert + i] != pred_labels[nvert + i]){
                    wrong_pred_test[i] = true;
                    wrong_pred_test_vlist.push_back(i);
                }

	    OUTPUT_MSG(__LINE__)

            vector<int> small_comp_wrongPred_vlist;
            vector<int> small_birth_wrongPred_vlist;
            vector<int> small_crit_wrongPred_vlist;
            vector<int> small_robCP_wrongPred_vlist;
            bool tmp_bool = false;
            for(i = 0; i < ntestv; ++i){
                if( ! wrong_pred_test[i] ) continue;
                tmp_bool = false;
                for(j = 0; j < nb_list_test2train[i].size(); ++j){
                    vid = nb_list_test2train[i][j];
                    tmp_bool = tmp_bool || small_comp_bool[vid];
                }
                if(tmp_bool)
                    small_comp_wrongPred_vlist.push_back(i);

                tmp_bool = false;
                for(j = 0; j < nb_list_test2train[i].size(); ++j){
                    vid = nb_list_test2train[i][j];
                    tmp_bool = tmp_bool || small_birth_bool[vid];
                }
                if(tmp_bool)
                    small_birth_wrongPred_vlist.push_back(i);
                 
                tmp_bool = false;
                for(j = 0; j < nb_list_test2train[i].size(); ++j){
                    vid = nb_list_test2train[i][j];
                    tmp_bool = tmp_bool || small_crit_bool[vid];
                }
                if(tmp_bool)
                    small_crit_wrongPred_vlist.push_back(i);
                 
                tmp_bool = false;
                for(j = 0; j < nb_list_test2train[i].size(); ++j){
                    vid = nb_list_test2train[i][j];
                    tmp_bool = tmp_bool || small_robCP_bool[vid];
                }
                if(tmp_bool)
                    small_robCP_wrongPred_vlist.push_back(i);
            }

	    OUTPUT_MSG(__LINE__)

            remove_redundant(small_comp_wrongPred_vlist);
            remove_redundant(small_birth_wrongPred_vlist);
            remove_redundant(small_crit_wrongPred_vlist);
            remove_redundant(small_robCP_wrongPred_vlist);
//            to_return->push_back(small_comp_wrongPred_vlist);
//            to_return->push_back(small_birth_wrongPred_vlist);
//            to_return->push_back(small_crit_wrongPred_vlist);
//            to_return->push_back(small_robCP_wrongPred_vlist);
        } // end of if(to_return != NULL)

	OUTPUT_MSG(__LINE__)

	return ret;
}

std::vector<std::vector<double> > PyPers (std::vector< double > vert_f, vector<vector<int> > knnG, int ntrain, double levelset_val, int skip1d_int, int curr_label, vector<int> true_labels, vector<int> pred_labels) {
 
#ifdef MATLAB
	fstream filestr;
	filestr.open (LOG_FILE, fstream::in | fstream::out | fstream::trunc);
	filestr << "Debuging Pers_2D" << endl;
	filestr.close();
#endif
	//mexWarnMsgTxt(LOG_FILE);
	
  double rob_thd = 0.0;
//  double levelset_val = 0 ;
  double perturb_thd = 0.0;

  int nv = vert_f.size();
// myassert( __LINE__ ,  nv == knnG.size() );
  myassert( __LINE__ ,  nv >= ntrain);
  myassert( __LINE__ ,  nv == knnG.size() );
  myassert( __LINE__ ,  nv == true_labels.size() );
  myassert( __LINE__ ,  nv == pred_labels.size() );

    bool skip1d = skip1d_int == 1;

    // vector<vector<double> > * persList = new vector<vector<double> >();
    vector<vector<double> > persList;
    // pers dots, each with dimension, birth vid, death vid, birth, death, robustness
        
    vector<bool> incorrect_pred_as_curr_label(nv, false); 
    // data that is incorrectly labeled as the current label
    for(int i = 0; i < nv; ++i){
        incorrect_pred_as_curr_label[i] = (curr_label != true_labels[i]) && (curr_label == pred_labels[i]);
    }

    vector<int> betti_numbers=calcPers(knnG,vert_f,ntrain, 
                rob_thd,levelset_val,skip1d, & persList, 
                incorrect_pred_as_curr_label);

        // cout << nv << " " << ne << " " << betti_numbers[0] << " " << betti_numbers[1] << " " << betti_numbers[2] << " " << betti_numbers[3] << " " << betti_numbers[4] << endl; 

	OUTPUT_MSG("calcPers is DONE");

        // prepare output
         
// 	myDoubleMatrix * persListM = new myDoubleMatrix(persList->size(), 6);     
//         for(int i = 0; i < persList->size(); ++i)
//             for(int j = 0; j < 6; ++j)
//                 persListM->data[i][j] = (* persList)[i][j];
// 
// 	    plhs[0] = mxCreateDoubleMatrix(1,betti_numbers.size(),mxREAL);
//             myDoubleMatrix::output1DIntArray(mxGetPr(plhs[0]),betti_numbers);
// 
//             plhs[1] = mxCreateDoubleMatrix(persList->size(), 6, mxREAL);
// 	    persListM->output1D(mxGetPr(plhs[1]));
// 
// 	    plhs[2] = mxCreateDoubleMatrix(final_edge_list->size(),2,mxREAL);
//             myDoubleMatrix::output2DDoubleArray(mxGetPr(plhs[2]),final_edge_list);
// 
//         delete persListM;

        vector<double> betti_numbers_double(betti_numbers.begin(), betti_numbers.end());
//        if(betti_numbers_double.size() < 6)
//            betti_numbers_double.resize(6);
        myassert( __LINE__ , betti_numbers_double.size() == 11);
        persList.insert(persList.begin(), betti_numbers_double);

	return persList;
}

std::vector<std::vector<double> > PyPersRev (std::vector< double > vert_f, vector<vector<int> > knnG, int ntrain, double levelset_val, int skip1d_int, int curr_label, vector<int> true_labels, vector<int> pred_labels) {
 
#ifdef MATLAB
	fstream filestr;
	filestr.open (LOG_FILE, fstream::in | fstream::out | fstream::trunc);
	filestr << "Debuging Pers_2D" << endl;
	filestr.close();
#endif
	//mexWarnMsgTxt(LOG_FILE);
	
  double rob_thd = 0.0;
//  double levelset_val = 0 ;
  double perturb_thd = 0.0;

  int nv = vert_f.size();
// myassert( __LINE__ ,  nv == knnG.size() );
  myassert( __LINE__ ,  nv >= ntrain);
  myassert( __LINE__ ,  nv == knnG.size() );
  myassert( __LINE__ ,  nv == true_labels.size() );
  myassert( __LINE__ ,  nv == pred_labels.size() );

    bool skip1d = skip1d_int == 1;

    // vector<vector<double> > * persList = new vector<vector<double> >();
    vector<vector<double> > persList;
    // pers dots, each with dimension, birth vid, death vid, birth, death, robustness
        
//    vector<bool> incorrect_pred_as_curr_label(nv, false); 
//    // data that is incorrectly labeled as the current label
//    for(int i = 0; i < nv; ++i){
//        incorrect_pred_as_curr_label[i] = (curr_label != true_labels[i]) && (curr_label == pred_labels[i]);
//    }

    vector<bool> incorrect_pred_as_other_label(nv, false); 
    // data that is incorrectly labeled as other labels
    for(int i = 0; i < nv; ++i){
        incorrect_pred_as_other_label[i] = (curr_label == true_labels[i]) && (curr_label != pred_labels[i]);
    }

    vector<int> betti_numbers=calcPers(knnG,vert_f,ntrain, 
                rob_thd,levelset_val,skip1d, & persList, 
                incorrect_pred_as_other_label);

        // cout << nv << " " << ne << " " << betti_numbers[0] << " " << betti_numbers[1] << " " << betti_numbers[2] << " " << betti_numbers[3] << " " << betti_numbers[4] << endl; 

	OUTPUT_MSG("calcPers is DONE");

        // prepare output
         
// 	myDoubleMatrix * persListM = new myDoubleMatrix(persList->size(), 6);     
//         for(int i = 0; i < persList->size(); ++i)
//             for(int j = 0; j < 6; ++j)
//                 persListM->data[i][j] = (* persList)[i][j];
// 
// 	    plhs[0] = mxCreateDoubleMatrix(1,betti_numbers.size(),mxREAL);
//             myDoubleMatrix::output1DIntArray(mxGetPr(plhs[0]),betti_numbers);
// 
//             plhs[1] = mxCreateDoubleMatrix(persList->size(), 6, mxREAL);
// 	    persListM->output1D(mxGetPr(plhs[1]));
// 
// 	    plhs[2] = mxCreateDoubleMatrix(final_edge_list->size(),2,mxREAL);
//             myDoubleMatrix::output2DDoubleArray(mxGetPr(plhs[2]),final_edge_list);
// 
//         delete persListM;

        vector<double> betti_numbers_double(betti_numbers.begin(), betti_numbers.end());
//        if(betti_numbers_double.size() < 6)
//            betti_numbers_double.resize(6);
        myassert( __LINE__ , betti_numbers_double.size() == 11);
        persList.insert(persList.begin(), betti_numbers_double);

	return persList;
}



std::vector<int> PyPersCC (std::vector< double > vert_f, vector<vector<int> > knnG, int ntrain, double levelset_val, int skip1d_int, int curr_label, vector<int> true_labels, vector<int> pred_labels) {
 
#ifdef MATLAB
	fstream filestr;
	filestr.open (LOG_FILE, fstream::in | fstream::out | fstream::trunc);
	filestr << "Debuging Pers_2D" << endl;
	filestr.close();
#endif
	//mexWarnMsgTxt(LOG_FILE);
	
  double rob_thd = 0.0;
//  double levelset_val = 0 ;
  double perturb_thd = 0.0;

  int nv = vert_f.size();
// myassert( __LINE__ ,  nv == knnG.size() );
  myassert( __LINE__ ,  nv >= ntrain);
  myassert( __LINE__ ,  nv == knnG.size() );
  myassert( __LINE__ ,  nv == true_labels.size() );
  myassert( __LINE__ ,  nv == pred_labels.size() );

    bool skip1d = skip1d_int == 1;

    // vector<vector<double> > * persList = new vector<vector<double> >();
    vector<vector<double> > persList;
    // pers dots, each with dimension, birth vid, death vid, birth, death, robustness
        
    vector<bool> incorrect_pred_as_curr_label(nv, false); 
    // data that is incorrectly labeled as the current label
    for(int i = 0; i < nv; ++i){
        incorrect_pred_as_curr_label[i] = (curr_label != true_labels[i]) && (curr_label == pred_labels[i]);
    }

    vector<int> incorr_comp_vert_list;
    vector<int> betti_numbers=calcPers(knnG,vert_f,ntrain, 
                rob_thd,levelset_val,skip1d, & persList, 
                incorrect_pred_as_curr_label, & incorr_comp_vert_list);
        // cout << nv << " " << ne << " " << betti_numbers[0] << " " << betti_numbers[1] << " " << betti_numbers[2] << " " << betti_numbers[3] << " " << betti_numbers[4] << endl; 

	OUTPUT_MSG("calcPers is DONE");

        // prepare output
         
// 	myDoubleMatrix * persListM = new myDoubleMatrix(persList->size(), 6);     
//         for(int i = 0; i < persList->size(); ++i)
//             for(int j = 0; j < 6; ++j)
//                 persListM->data[i][j] = (* persList)[i][j];
// 
// 	    plhs[0] = mxCreateDoubleMatrix(1,betti_numbers.size(),mxREAL);
//             myDoubleMatrix::output1DIntArray(mxGetPr(plhs[0]),betti_numbers);
// 
//             plhs[1] = mxCreateDoubleMatrix(persList->size(), 6, mxREAL);
// 	    persListM->output1D(mxGetPr(plhs[1]));
// 
// 	    plhs[2] = mxCreateDoubleMatrix(final_edge_list->size(),2,mxREAL);
//             myDoubleMatrix::output2DDoubleArray(mxGetPr(plhs[2]),final_edge_list);
// 
//         delete persListM;

        vector<double> betti_numbers_double(betti_numbers.begin(), betti_numbers.end());
//        if(betti_numbers_double.size() < 6)
//            betti_numbers_double.resize(6);
        myassert( __LINE__ , betti_numbers_double.size() == 11);
        persList.insert(persList.begin(), betti_numbers_double);

        // return persList;
	return incorr_comp_vert_list;
}

std::vector<int> PyPersCCRev (std::vector< double > vert_f, vector<vector<int> > knnG, int ntrain, double levelset_val, int skip1d_int, int curr_label, vector<int> true_labels, vector<int> pred_labels) {
 
#ifdef MATLAB
	fstream filestr;
	filestr.open (LOG_FILE, fstream::in | fstream::out | fstream::trunc);
	filestr << "Debuging Pers_2D" << endl;
	filestr.close();
#endif
	//mexWarnMsgTxt(LOG_FILE);
	
  double rob_thd = 0.0;
//  double levelset_val = 0 ;
  double perturb_thd = 0.0;

  int nv = vert_f.size();
// myassert( __LINE__ ,  nv == knnG.size() );
  myassert( __LINE__ ,  nv >= ntrain);
  myassert( __LINE__ ,  nv == knnG.size() );
  myassert( __LINE__ ,  nv == true_labels.size() );
  myassert( __LINE__ ,  nv == pred_labels.size() );

    bool skip1d = skip1d_int == 1;

    // vector<vector<double> > * persList = new vector<vector<double> >();
    vector<vector<double> > persList;
    // pers dots, each with dimension, birth vid, death vid, birth, death, robustness
        
//    vector<bool> incorrect_pred_as_curr_label(nv, false); 
//    // data that is incorrectly labeled as the current label
//    for(int i = 0; i < nv; ++i){
//        incorrect_pred_as_curr_label[i] = (curr_label != true_labels[i]) && (curr_label == pred_labels[i]);
//    }

    vector<bool> incorrect_pred_as_other_label(nv, false); 
    // data that is incorrectly labeled as other labels
    for(int i = 0; i < nv; ++i){
        incorrect_pred_as_other_label[i] = (curr_label == true_labels[i]) && (curr_label != pred_labels[i]);
    }


    vector<int> incorr_comp_vert_list;
    vector<int> betti_numbers=calcPers(knnG,vert_f,ntrain, 
                rob_thd,levelset_val,skip1d, & persList, 
                incorrect_pred_as_other_label, & incorr_comp_vert_list);
        // cout << nv << " " << ne << " " << betti_numbers[0] << " " << betti_numbers[1] << " " << betti_numbers[2] << " " << betti_numbers[3] << " " << betti_numbers[4] << endl; 

	OUTPUT_MSG("calcPers is DONE");

        // prepare output
         
// 	myDoubleMatrix * persListM = new myDoubleMatrix(persList->size(), 6);     
//         for(int i = 0; i < persList->size(); ++i)
//             for(int j = 0; j < 6; ++j)
//                 persListM->data[i][j] = (* persList)[i][j];
// 
// 	    plhs[0] = mxCreateDoubleMatrix(1,betti_numbers.size(),mxREAL);
//             myDoubleMatrix::output1DIntArray(mxGetPr(plhs[0]),betti_numbers);
// 
//             plhs[1] = mxCreateDoubleMatrix(persList->size(), 6, mxREAL);
// 	    persListM->output1D(mxGetPr(plhs[1]));
// 
// 	    plhs[2] = mxCreateDoubleMatrix(final_edge_list->size(),2,mxREAL);
//             myDoubleMatrix::output2DDoubleArray(mxGetPr(plhs[2]),final_edge_list);
// 
//         delete persListM;

        vector<double> betti_numbers_double(betti_numbers.begin(), betti_numbers.end());
//        if(betti_numbers_double.size() < 6)
//            betti_numbers_double.resize(6);
        myassert( __LINE__ , betti_numbers_double.size() == 11);
        persList.insert(persList.begin(), betti_numbers_double);

        // return persList;
	return incorr_comp_vert_list;
}

vector<vector<int> > PyPersAll (std::vector< double > vert_f, vector<vector<int> > knnG, int ntrain, double levelset_val, int skip1d_int, int curr_label, vector<int> true_labels, vector<int> pred_labels) {
 
#ifdef MATLAB
	fstream filestr;
	filestr.open (LOG_FILE, fstream::in | fstream::out | fstream::trunc);
	filestr << "Debuging Pers_2D" << endl;
	filestr.close();
#endif
	//mexWarnMsgTxt(LOG_FILE);
	
  double rob_thd = 0.0;
//  double levelset_val = 0 ;
  double perturb_thd = 0.0;

  int nv = vert_f.size();
// myassert( __LINE__ ,  nv == knnG.size() );
  myassert( __LINE__ ,  nv >= ntrain);
  myassert( __LINE__ ,  nv == knnG.size() );
  myassert( __LINE__ ,  nv == true_labels.size() );
  myassert( __LINE__ ,  nv == pred_labels.size() );

    bool skip1d = skip1d_int == 1;

    // vector<vector<double> > * persList = new vector<vector<double> >();
    vector<vector<double> > persList;
    // pers dots, each with dimension, birth vid, death vid, birth, death, robustness
        
    vector<bool> incorrect_pred_as_curr_label(nv, false); 
    // data that is incorrectly labeled as the current label
    for(int i = 0; i < nv; ++i){
        incorrect_pred_as_curr_label[i] = (curr_label != true_labels[i]) && (curr_label == pred_labels[i]);
    }

    vector<int> incorr_comp_vert_list;
    vector<vector<int> > to_return;
    vector<int> betti_numbers=calcPers(knnG,vert_f,ntrain, 
                rob_thd,levelset_val,skip1d, & persList, 
                incorrect_pred_as_curr_label, & incorr_comp_vert_list,
                true_labels, pred_labels, & to_return);
        // cout << nv << " " << ne << " " << betti_numbers[0] << " " << betti_numbers[1] << " " << betti_numbers[2] << " " << betti_numbers[3] << " " << betti_numbers[4] << endl; 

	OUTPUT_MSG("calcPers is DONE");

        vector<double> betti_numbers_double(betti_numbers.begin(), betti_numbers.end());
        myassert( __LINE__ , betti_numbers_double.size() == 11);
        persList.insert(persList.begin(), betti_numbers_double);

	return to_return;
}


PYBIND11_PLUGIN(PythonGraphPers_withCompInfo) {
        py::module m("PythonGraphPers_withCompInfo", "python binding for persistence computation (cubical complex)");

//    m.def("kw_func4", &kw_func4, py::arg("myList") = list);
    m.def("PyPers", &PyPers);
    m.def("PyPersRev", &PyPersRev);
    m.def("PyPersCC", &PyPersCC);
    m.def("PyPersCCRev", &PyPersCCRev);
    m.def("PyPersAll", &PyPersAll);
    return m.ptr();
}
