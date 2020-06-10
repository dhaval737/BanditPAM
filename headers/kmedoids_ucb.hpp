/**
 * @file kmedoids_ucb.hpp
 * @date 2020-06-08
 *
 * This header file specifies the implementation of a KMedoids algorithm
 * that makes use of theory specified in a paper submitted to NeurIPS
 */

#include <armadillo>
#include <fstream>
#include <iostream>
#include <omp.h>

class KMediods {
  public:
    /**
     * @brief Construct a new KMediods object
     *
     * @param maxIterations the maximum number of iterations for the swap step
     */
    KMediods(size_t maxIterations = 1000);

    /**
     * @brief main method for kmedoids algorithm. Builds initial medoids then
     * calls the swap method and iterates until convergence.
     *
     * @param data column major dataset that the kmedoids algorithm will be run
     * on
     * @param clusters number of clusters to fit data to
     * @param assignments vector which indicates the cluster that data point i
     * belonds to
     * @param medoid_indicies vector where indices(i) is the medoid of cluster i
     */
    void cluster(const arma::mat &data, const size_t clusters,
                 arma::urowvec &assignments, arma::urowvec &medoid_indicies);

  private:
    /**
     * @brief build the sigma vector for the build step
     *
     * This method is called before a medoid is built, and calculates
     * the standard deviation of the loss selected from a number of
     * temporary reference points. On the first iteration the absolute
     * loss is used, and on all subsequent iterations the difference in loss
     * is used.
     *
     * @param data column major dataset that the kmedoids algorithm will be run
     * on
     * @param best_distances vector containing the minimum distance between
     * point i and a medoid
     * @param sigma vector containing the sigmas for setting point i to be the
     * next medoid
     * @param batch_size batch size used in calcuating sigma
     * @param use_absolute boolean valute indicating whether to use absolute
     * loss or difference of loss. Set to true on first iteration, false after.
     */
    void build_sigma(const arma::mat &data, arma::rowvec &best_distances,
                     arma::rowvec &sigma, arma::uword batch_size,
                     bool use_absolute);

    /**
     * @brief calculate an estimation of loss for targets
     *
     * This method calculates an estimate for the loss of the points specified
     * in target using batch_size number of points.
     *
     * @param data column major dataset that the kmedoids algorithim will be run
     * on
     * @param target points to calculate an estimation of loss for
     * @param batch_size number of points to use in calculating estimate of loss
     * @param best_distances vector containing the minimum distance between
     * point i and a medoid
     * @param use_absolute boolean valute indicating whether to use absolute
     * loss or difference of loss. Set to true on first iteration, false after.
     * @return arma::rowvec containing the estimates of the loss for the target
     * points
     */
    arma::rowvec build_target(const arma::mat &data, arma::uvec &target,
                              size_t batch_size, arma::rowvec &best_distances,
                              bool use_absolute);

    /**
     * @brief Select the initial medoids using a greedy initialization
     *
     * @param data column major dataset that the kmedoids algorithim will be run
     * on
     * @param clusters number of clusters to select
     * @param medoid_indicies indicies in data of the selected medoids
     * @param medoids matrix containing the selected medoids in column major
     * format
     */
    void build(const arma::mat &data, const size_t clusters,
               arma::urowvec &medoid_indicies, arma::mat &medoids);

    double cost_fn_build(const arma::mat &data, arma::uword target,
                         arma::uvec &tmp_refs, arma::rowvec &best_distances);

    /**
     * @brief calculate estimates of the difference in loss for targets in the
     * swap step
     *
     * @param data column major dataset that the kmedoids algorithim will be run
     * on
     * @param medoid_indices indices of medoids
     * @param targets targets to calculate estimate of loss difference for
     * @param batch_size number of reference points to use to estimate loss
     * @param best_distances minimum distance between point i and a medoid
     * @param second_best_distances second smallest distance between point i and
     * a medoid
     * @param assignments vector containing the current cluster assignments for
     * each data point
     * @return arma::vec containing the estimates of loss difference for the
     * specified targets.
     */
    arma::vec swap_target(const arma::mat &data, arma::urowvec &medoid_indices,
                          arma::uvec &targets, size_t batch_size,
                          arma::rowvec &best_distances,
                          arma::rowvec &second_best_distances,
                          arma::urowvec &assignments);

    /**
     * @brief iteratively swap medoids until convergence or maxIterations
     *
     * @param data column major dataset that the kmedoids algorithim will be run
     * on
     * @param clusters number of clusters to fit data to
     * @param medoid_indicies indiceies of data points that are serving as
     * medoids
     * @param medoids column major matrix of medoids
     */
    void swap(const arma::mat &data, const size_t clusters,
              arma::urowvec &medoid_indicies, arma::mat &medoids, arma::urowvec& assignments);

    /**
     * @brief debugging function for checking absolute loss of some given
     * medoid_indices
     *
     * @param data column major dataset
     * @param clusters number of clusters to use
     * @param medoid_indicies vector containing indicies of medoids
     * @return double indicating total loss
     */
    double calc_loss(const arma::mat &data, const size_t clusters,
                     arma::urowvec &medoid_indicies);

    /**
     * @brief calculate sigma for swap iteration
     *
     * @param data column major dataset
     * @param sigma vector of std deviation of loss difference for each
     * iteration
     * @param batch_size number of reference points to use in estimate of loss
     * difference
     * @param best_distances minimum distance between a point and and medoid
     * @param second_best_distances second best distance between a point and any
     * medoid
     * @param assignments vector containing the current cluster assignments for
     * each data point
     */
    void swap_sigma(const arma::mat &data, arma::mat &sigma, size_t batch_size,
                    arma::rowvec &best_distances,
                    arma::rowvec &second_best_distances,
                    arma::urowvec &assignments);

    double sigma_const = 0.1;
    size_t maxIterations;
    int verbosity = 0;
};