#include <omp.h>
#include <random>
#include <cmath>
#include "ncvis.hpp"
#include "../lib/pcg-cpp/include/pcg_random.hpp"
#include <chrono>
#include <iostream>
#include <fstream>
#include <map>
#include <string>

ncvis::NCVis::NCVis(long d, long n_threads, long n_neighbors, long M,
                    long ef_construction, long random_seed, int n_epochs, 
                    int n_init_epochs, float a, float b, float alpha, float alpha_Q,
                    long* n_noise, ncvis::Distance dist, bool fix_Q, bool noise_in_ratio,
                    float noise_in_ratio_val, bool learn_Q):
d_(d), M_(M), ef_construction_(ef_construction), 
random_seed_(random_seed), n_neighbors_(n_neighbors), n_epochs_(n_epochs),
n_init_epochs_(n_init_epochs), a_(a), b_(b), alpha_(alpha), alpha_Q_(alpha_Q),
space_(nullptr), appr_alg_(nullptr), dist_(dist),
fix_Q_(fix_Q), noise_in_ratio_(noise_in_ratio), noise_in_ratio_val_(noise_in_ratio_val),
learn_Q_(learn_Q), precomp_init(false)
{
    omp_set_num_threads(n_threads);
    n_noise_ = new long[n_epochs];
    long default_noise = 3;
    for (int i=0; i<n_epochs; ++i){
        n_noise_[i] = (n_noise == nullptr)? default_noise:n_noise[i];
    }
}

ncvis::NCVis::~NCVis(){
    delete space_;
    space_ = nullptr;
    delete appr_alg_;
    appr_alg_ = nullptr;
    delete[] n_noise_;
    n_noise_ =  nullptr;
}

void ncvis::NCVis::preprocess(const float *const x, long D, ncvis::Distance dist, float* out){
    if (dist == ncvis::Distance::correlation){
        float M = 0;
        for (long i = 0; i < D; ++i){
            M += x[i];
        }
        M /= D;
        // printf("[ncvis::NCVis::preprocess] M = %f\n", M);
        for (long i = 0; i < D; ++i){
            out[i] = x[i]-M;
        }
    }
    else{
        for (long i = 0; i < D; ++i){
            out[i] = x[i];
        }
    }
    // printf("[ncvis::NCVis::preprocess](center) [");
    // for (long j=0; j<D; ++j){
    //     printf("%5.1f ", out[j]);
    // }
    // printf("]\n");
    if (dist == ncvis::Distance::correlation || dist == ncvis::Distance::cosine_similarity){
        float N = 0;
        for (long i = 0; i < D; ++i){
            N += out[i]*out[i];
        }
        // printf("[ncvis::NCVis::preprocess] N = %f\n", sqrtf(N));
        if (N != 0){
            for (long i = 0; i < D; ++i){
                out[i] /= sqrtf(N);
            }
        }
    }
    // printf("[ncvis::NCVis::preprocess](norm) [");
    // for (long j=0; j<D; ++j){
    //     printf("%5.1f ", out[j]);
    // }
    // printf("]\n");
}

void ncvis::NCVis::buildKNN(const float *const X, long N, long D){
    delete space_;
    space_ = nullptr;
    delete appr_alg_;
    appr_alg_ = nullptr;

    switch(dist_){
        case ncvis::Distance::squared_L2:
            // printf("[ncvis::NCVis::buildKNN] squared_L2\n");
            space_ = new hnswlib::L2Space(D);
            break;
        case ncvis::Distance::inner_product:
        case ncvis::Distance::cosine_similarity:
        case ncvis::Distance::correlation:
            // printf("[ncvis::NCVis::buildKNN] inner_product || cosine_similarity || correlation\n");
            space_ = new hnswlib::InnerProductSpace(D);
            break;
        default: 
            throw std::runtime_error("[ncvis::NCVis::buildKNN] Unrecognized distance type.");
            break;
    }
    appr_alg_ = new hnswlib::HierarchicalNSW<float>(space_, N, M_, ef_construction_, random_seed_);

    // Perform initialisation without messing with mutexes 
    float* x = new float[D];
    preprocess(X, D, dist_, x);
    appr_alg_->addPoint((void*)x, 0);
    delete[] x;

    #pragma omp parallel
    {
    float* x = new float[D];
    #pragma omp for
    for (long i=1; i < N; ++i){
        // printf("[%lu]>> [", i);
        // for (long j=0; j<D; ++j){
        //     printf("%5.1f ", X[j+D*i]);
        // }
        // printf("]\n");
        preprocess(X+i*D, D, dist_, x);
        // printf("[%lu]<< [", i);
        // for (long j=0; j<D; ++j){
        //     printf("%5.1f ", x[j]);
        // }
        // printf("]\n");
        appr_alg_->addPoint((void*)x, i);
    }
    delete[] x;
    }
}

ncvis::KNNTable ncvis::NCVis::findKNN(const float *const X, long N, long D, long k){
    KNNTable table(N, k);

    #pragma omp parallel
    {
    float* x = new float[D];
    #pragma omp for
    for (long i=0; i < N; ++i){
        // Find k+1 neighbors as one of them is the point itself
        preprocess(X+i*D, D, dist_, x);
        auto result = appr_alg_->searchKnn((const void*)x, k+1);
        if ((long)result.size() != k+1){
            std::cout << "[ncvis::NCVis::findKNN] Found less than k nearest neighbors, try increasing M or ef_construction.";
        } else{
            for (long j=0; j<k; ++j) {
                auto& result_tuple = result.top();
                table.dists[i].push_back(result_tuple.first);
                table.inds[i].push_back(result_tuple.second);
                result.pop();
            }
        }
    }
    delete[] x;
    }
    return table;
}

std::vector<ncvis::Edge> ncvis::NCVis::build_edges(ncvis::KNNTable& table){
    long n_edges = 0;

    #pragma omp parallel for
    for (long i = 0; i < (long)table.inds.size(); ++i){
        #pragma omp atomic
        n_edges += (long)table.inds[i].size();
    }
    std::vector<ncvis::Edge> edges;
    edges.reserve(n_edges);

    for (long i = 0; i < (long)table.inds.size(); ++i){
        for (long j =0; j < (long)table.inds[i].size(); ++j){
            edges.emplace_back(i, table.inds[i][j]);
        }
    }

    return edges;
}

float ncvis::NCVis::d_sqr(const float *const x, const float *const y){
    float dist_sqr = 0;
    for (long i = 0; i < d_; ++i){
        dist_sqr += (x[i]-y[i])*(x[i]-y[i]);
    }

    return dist_sqr;
}

void ncvis::NCVis::init_embedding(long N, float* Y, float alpha, std::vector<ncvis::Edge>& edges){
    // For temporary values
    float* Ys[2];
    Ys[0] = Y;
    Ys[1] = new float[N*d_];
    float* mean = new float[d_];
    float* sigma = new float[d_];

    #pragma omp parallel
    {
    int id = omp_get_thread_num();
    pcg64 pcg(random_seed_ + id);
    std::uniform_real_distribution<float> gen_Y(0, 1);

    #pragma omp for
    for (long i = 0; i < N*d_; ++i){
        Y[i] = gen_Y(pcg);
    }
    // Initialize layout
    for (int init_epoch = 0; init_epoch < n_init_epochs_; ++init_epoch){
        float* Y_old = Ys[init_epoch % 2];
        float* Y_new = Ys[(init_epoch+1) % 2];
        #pragma omp for
        for (long i = 0; i < N*d_; ++i){
            Y_new[i] = 0;
        }
        
        #pragma omp for
        for (long i = 0; i < (long)edges.size(); ++i){
            long id = edges[i].first;
            long other_id = edges[i].second;
            for (long k = 0; k < d_; ++k){
                Y_new[id*d_+k] += alpha * Y_old[other_id*d_+k];
            }
        }

        #pragma omp single
        for (long k = 0; k < d_; ++k){
            mean[k] = 0;
            sigma[k] = 0;
        }

        #pragma omp for
        for (long i = 0; i < N; ++i){
            for (long k = 0; k < d_; ++k){
                #pragma omp atomic
                mean[k] += Y_new[i*d_+k];
            }
        }

        #pragma omp single
        for (long k = 0; k < d_; ++k){
            mean[k] /= N;
        }
    
        #pragma omp for
        for (long i = 0; i < N; ++i){
            for (long k = 0; k < d_; ++k){
                float delta2 = Y_new[i*d_+k] - mean[k];
                delta2 *= delta2;
                #pragma omp atomic
                sigma[k] += delta2;
            }
        }

        #pragma omp single
        for (long k = 0; k < d_; ++k){
            sigma[k] /= N;
            sigma[k] = sqrtf(sigma[k]);
        }
        
        #pragma omp for
        for (long i = 0; i < N; ++i){
            for (long k = 0; k < d_; ++k){
                Y_new[i*d_+k] = (Y_new[i*d_+k] - mean[k])/sigma[k];
            }
        }
    }
    if (n_init_epochs_ % 2){
        #pragma omp for
        for (long i = 0; i < N*d_; ++i){
            Ys[0][i] = Ys[1][i];
        }
    }
    }
    
    delete[] Ys[1];
    delete[] mean;
    delete[] sigma;
}

//std::pair<std::vector<float>, std::vector<std::vector<float>>> ncvis::NCVis::optimize(long N, float* Y, float& Q, std::vector<ncvis::Edge>& edges){
std::map<std::string, std::vector<float>> ncvis::NCVis::optimize(long N, float* Y, float& Q, std::vector<ncvis::Edge>& edges){

    float Q_cum=0.;
    //#pragma omp parallel
    //{
    int id = omp_get_thread_num();
    int n_threads = omp_get_num_threads();
    pcg64 pcg(random_seed_+id);
    // Build layout
    std::uniform_int_distribution<long> gen_ind(0, N-1);

    std::vector<float> Q_data;
    std::vector<float> E_data;

    // copy initial values of Y to vector to return them
    std::vector<float> Y_epoch{Y, Y+N*d_};
    E_data.insert(E_data.end(), Y_epoch.begin(), Y_epoch.end());
    Q_data.push_back(Q);

    // create degree vector
    std::vector<float> deg(N, 0.0);
    if (noise_in_ratio_){
        for (long i = 0; i < (long)edges.size(); ++i){
            deg[edges[i].first] += 1.0; // only count out-degree
            // deg[edges[i].second] += 1.0; // every edge appears twice in edges (once in each direction), but we only want to consider it with weight 1.
        }
    }

    for (int epoch = 0; epoch < n_epochs_; ++epoch){
        float step = alpha_*(1-(((float)epoch)/n_epochs_)*(((float)epoch)/n_epochs_));
        float step_Q = alpha_Q_;
        if (fix_Q_){
            // damped just like for the step size of the embeddings
             step_Q *= (1-(((float)epoch)/n_epochs_)*(((float)epoch)/n_epochs_));
        }
        float Q_copy = Q;
        if (!learn_Q_){
            Q_copy = 0.;
        }
        long cur_noise = n_noise_[epoch];
        #pragma omp for nowait
        for (long i = 0; i < (long)edges.size(); ++i){
            long id = edges[i].first;
            for (long j = 0; j < cur_noise+1; ++j){
                long other_id;
                if (j == 0){
                    other_id = edges[i].second; 
                } else{
                    do{
                        other_id = gen_ind(pcg);
                    } while (other_id == id);
                }

                float d2 = d_sqr(Y+id*d_, Y+other_id*d_);
                float Ph = 1/(1+a_*powf(d2, b_));
                float w = 1.;
                if (cur_noise != 0){
                    if (noise_in_ratio_){
                        float prob_noise = deg[id]/(((float)N-1.0)*(float)edges.size());
                        w = Ph/(cur_noise*prob_noise*expf(Q_copy));
                    } else {
                        w = Ph/(noise_in_ratio_val_*expf(Q_copy));
                        // w = Ph/(cur_noise*expf(Q));
                    }

                    if (j == 0){
                        w = 1/(1+w);    
                    } else {
                        w = -1/(1+1/w);
                    }

                    if (learn_Q_){
                        Q_copy -= w*step_Q;
                    }

                    w = 2*w*Ph*a_*b_*powf(d2, b_-1);
                }
                for (long k = 0; k < d_; ++k){
                    float dx_k = Y[other_id*d_+k] - Y[id*d_+k];
                    dx_k = dx_k * w * step;
                    if (dx_k > 4.){
                        dx_k = 4.;
                    } else if (dx_k < -4.){
                        dx_k = -4.;
                    }
                    Y[id*d_+k] += dx_k;
                    Y[other_id*d_+k] -= dx_k;
                }
            }  
        }
        #pragma omp atomic
        Q_cum += Q_copy;
        #pragma omp barrier
        #pragma omp single
        {
        Q = Q_cum/n_threads;
        Q_cum = 0;

        Q_data.push_back(Q);

        // copy current values of Y to vector to return them
        std::vector<float> Y_epoch{Y, Y+N*d_};
        E_data.insert(E_data.end(), Y_epoch.begin(), Y_epoch.end());
        }
    }

    std::map<std::string, std::vector<float>> aux_data { {"qs", Q_data},
                                                         {"embds", E_data} };
    return aux_data;
}

// version in which edges are computed
std::map<std::string, std::vector<float>> ncvis::NCVis::fit_transform(const float *const X, long N, long D, float* Y){
    // printf("==============DATA============\n");
    // for (long i=0; i<N; ++i){
    //     printf("[");
    //     for (long j=0; j<D; ++j){
    //         printf("%5.2f ", X[j+D*i]);
    //     }
    //     printf("]\n");
    // }
    // printf("===============================\n");
    if (N == 0 || D == 0){
        throw std::runtime_error("[ncvis::NCVis::fit_transform] Dataset should have at least one element.");
    }
    if (Y == nullptr){
        throw std::runtime_error("[ncvis::NCVis::fit_transform] Null pointer provided for output.");
    }
    #if defined(DEBUG)
        auto t1 = std::chrono::high_resolution_clock::now();
    #endif
    buildKNN(X, N, D);
    #if defined(DEBUG)
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "buildKNN: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                  << " ms\n";
    #endif
    // Number of neighbors can't exceed the total number of other points 
    long k = (n_neighbors_ < N-1)? n_neighbors_:(N-1);
    k = (k > 0)? k:1;
    #if defined(DEBUG)
        t1 = std::chrono::high_resolution_clock::now();
    #endif
    KNNTable table = findKNN(X, N, D, k);

    // The graph itself is no longer needed
    delete appr_alg_;
    appr_alg_ = nullptr;
    delete space_;
    space_ = nullptr;

    #if defined(DEBUG)
        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "findKNN: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                << " ms\n";
        t1 = std::chrono::high_resolution_clock::now();
    #endif
    table.symmetrize();

    #if defined(DEBUG)
        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "symmetrize: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                << " ms\n";
        t1 = std::chrono::high_resolution_clock::now();
    #endif
    std::vector<ncvis::Edge> edges = build_edges(table);

    #if defined(DEBUG)
        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "build_edges: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                << " ms\n";
    #endif
    // Normalization
    float Q=0.;
    float init_alpha = 1./k;

    #if defined(DEBUG)
        t1 = std::chrono::high_resolution_clock::now();
    #endif

    // if no precomputed initialization was given, compute one
    if (!precomp_init){
        init_embedding(N, Y, init_alpha, edges);
    }

    #if defined(DEBUG)
        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "initialize: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                << " ms\n";
        t1 = std::chrono::high_resolution_clock::now();
    #endif
    std::map<std::string, std::vector<float>> aux_data = optimize(N, Y, Q, edges);
    #if defined(DEBUG)
        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "optimize: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                << " ms\n";
    #endif
    // printf("============DISTANCES==========\n");
    // for (long i=0; i<N; ++i){
    //     printf("[");
    //     for (long j=0; j<table.dists[i].size(); ++j){
    //         printf("%f ", table.dists[i][j]);
    //     }
    //     printf("]\n");
    // }
    // printf("===============================\n");

    // printf("============NEIGHBORS==========\n");
    // for (long i=0; i<N; ++i){
    //     printf("[");
    //     for (long j=0; j<table.inds[i].size(); ++j){
    //         printf("%ld ", table.inds[i][j]);
    //     }
    //     printf("]\n");
    // }
    // printf("===============================\n");
    //return data;

    // transfrom edges to float for logging
    std::vector<float> edges_float;
    for (long i = 0; i < (long)edges.size(); ++i){
        edges_float.push_back((float)edges[i].first);
        edges_float.push_back((float)edges[i].second);
    }
    aux_data["edges"] = edges_float;

    return aux_data;
}


// version in which edges are given
std::map<std::string, std::vector<float>> ncvis::NCVis::fit_transform_edges(const float *const X, long N, long D, float* Y, long* precomp_edges, long n_edges){
    // printf("==============DATA============\n");
    // for (long i=0; i<N; ++i){
    //     printf("[");
    //     for (long j=0; j<D; ++j){
    //         printf("%5.2f ", X[j+D*i]);
    //     }
    //     printf("]\n");
    // }
    // printf("===============================\n");
    if (N == 0 || D == 0){
        throw std::runtime_error("[ncvis::NCVis::fit_transform] Dataset should have at least one element.");
    }
    if (Y == nullptr){
        throw std::runtime_error("[ncvis::NCVis::fit_transform] Null pointer provided for output.");
    }
    #if defined(DEBUG)
        auto t1 = std::chrono::high_resolution_clock::now();
    #endif
    // Number of neighbors can't exceed the total number of other points
    long k = (n_neighbors_ < N-1)? n_neighbors_:(N-1);
    k = (k > 0)? k:1;


    // bring precomputed edges into the right format
    std::vector<ncvis::Edge> edges;
    edges.reserve(n_edges);

    for (long i = 0; i < (long)n_edges; ++i){
        edges.emplace_back(precomp_edges[2*i], precomp_edges[2*i + 1]);
    }


    // Normalization
    float Q=0.;
    float init_alpha = 1./k;

    #if defined(DEBUG)
        t1 = std::chrono::high_resolution_clock::now();
    #endif

    // if no precomputed initialization was given, compute one
    if (!precomp_init){
        init_embedding(N, Y, init_alpha, edges);
    }

    #if defined(DEBUG)
        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "initialize: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                << " ms\n";
        t1 = std::chrono::high_resolution_clock::now();
    #endif
    std::map<std::string, std::vector<float>> aux_data = optimize(N, Y, Q, edges);
    #if defined(DEBUG)
        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "optimize: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                << " ms\n";
    #endif
    // printf("============DISTANCES==========\n");
    // for (long i=0; i<N; ++i){
    //     printf("[");
    //     for (long j=0; j<table.dists[i].size(); ++j){
    //         printf("%f ", table.dists[i][j]);
    //     }
    //     printf("]\n");
    // }
    // printf("===============================\n");

    // printf("============NEIGHBORS==========\n");
    // for (long i=0; i<N; ++i){
    //     printf("[");
    //     for (long j=0; j<table.inds[i].size(); ++j){
    //         printf("%ld ", table.inds[i][j]);
    //     }
    //     printf("]\n");
    // }
    // printf("===============================\n");
    //return data;

    // transfrom edges to float for logging
    std::vector<float> edges_float;
    for (long i = 0; i < (long)edges.size(); ++i){
        edges_float.push_back((float)edges[i].first);
        edges_float.push_back((float)edges[i].second);
    }
    aux_data["edges"] = edges_float;


    return aux_data;
}
