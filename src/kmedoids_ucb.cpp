#include "kmedoids_ucb.hpp"

using namespace arma;

KMediods::KMediods(size_t maxIterations)
{
    this->maxIterations = maxIterations;
}

void KMediods::cluster(const arma::mat &data,
                       const size_t clusters,
                       arma::Row<size_t> &assignments)
{
    arma::mat medoids(data.n_rows, clusters);
    arma::Row<size_t> medoid_indicies(clusters);

    // build clusters
    KMediods::build(data, clusters, medoid_indicies, medoids);

    std::cout << medoid_indicies << std::endl;
    std::cout << "########################### SWAP ##############################" << std::endl;

    KMediods::swap(data, clusters, medoid_indicies, medoids);
    std::cout << medoid_indicies << std::endl;
}

void KMediods::build(const arma::mat &data,
                     const size_t clusters,
                     arma::Row<size_t> &medoid_indicies,
                     arma::mat &medoids)
{
    // Parameters
    size_t N = data.n_cols;
    arma::Row<size_t> N_mat(N);
    N_mat.fill(N);
    double p = 1 / (N * 10);
    bool use_absolute = true;
    arma::Row<size_t> num_samples(N, arma::fill::zeros);
    arma::Row<double> estimates(N, arma::fill::zeros);

    arma::Row<double> best_distances(N);
    arma::rowvec sigma(N);

    best_distances.fill(std::numeric_limits<double>::infinity());
    //best_distances.fill(1000000000);

    for (size_t k = 0; k < clusters; k++)
    {
        size_t step_count = 0;
        arma::urowvec candidates(N, arma::fill::ones); // one hot encoding of candidates;
        arma::Row<double> lcbs(N);
        arma::Row<double> ucbs(N);
        //lcbs.fill(1000); //Is it safe to make the assumption that something will always be overwriting this?
        //ucbs.fill(1000);
        arma::Row<size_t> T_samples(N, arma::fill::zeros);
        arma::Row<size_t> exact_mask(N, arma::fill::zeros);

        size_t original_batch_size = 20;

        //sigma.fill(.1);
        std::cout << "filling the sigma " << std::endl;
        KMediods::build_sigma(data, best_distances, sigma, original_batch_size, use_absolute);
        std::cout << "filled the sigma" << std::endl;

        size_t base = 1;

        while (arma::sum(candidates) > 0.1)
        {
            std::cout << "Step count" << step_count << std::endl;
            size_t this_batch_size = original_batch_size; //need to add scaling batch size

            arma::umat compute_exactly = (T_samples + this_batch_size) >= N_mat;

            compute_exactly = compute_exactly != exact_mask; //check this
            if (arma::accu(compute_exactly) > 0)
            {
                uvec targets = find(compute_exactly);
                //std::cout << "Computing exactly for " << targets.n_rows << " on step count " << step_count << std::endl;

                arma::Row<double> result = build_target(data, targets, N, best_distances, use_absolute);
                //std::cout << "setting estimates" << std::endl;

                estimates.cols(targets) = result;
                //std::cout << "setting ucbs" << std::endl;

                ucbs.cols(targets) = result;
                //std::cout << "setting lcbs" << std::endl;

                lcbs.cols(targets) = result;

                exact_mask.cols(targets).fill(1);
                T_samples.cols(targets) += N;
                candidates.cols(targets).fill(0);
            }

            if (sum(candidates) < 0.5)
            {
                continue;
            }
            uvec targets = find(candidates);
            arma::Row<double> result = build_target(data, targets, this_batch_size, best_distances, use_absolute);
            estimates.cols(targets) = ((T_samples.cols(targets) % estimates.cols(targets)) + (result * this_batch_size)) / (this_batch_size + T_samples.cols(targets));
            T_samples.cols(targets) += this_batch_size;
            arma::Row<double> adjust(targets.n_rows);
            adjust.fill(std::log(1 / p));
            //arma::Row<double> cb_delta = sigma.cols(targets) % arma::sqrt(adjust / T_samples.cols(targets));
            arma::Row<double> cb_delta = 0.1 * arma::sqrt(adjust / T_samples.cols(targets));

            ucbs.cols(targets) = estimates.cols(targets) + cb_delta;
            lcbs.cols(targets) = estimates.cols(targets) - cb_delta;

            candidates = (lcbs < ucbs.min()) != exact_mask;
            step_count++;
        }

        arma::uword new_medoid = lcbs.index_min();
        medoid_indicies(k) = lcbs.index_min();
        medoids.col(k) = data.col(medoid_indicies(k));

        // don't need to do this on final iteration
        //multithread, decompose
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j <= k; j++)
            {
                double cost = norm(data.col(i) - data.col(medoid_indicies(j)), 2);
                if (cost < best_distances(i))
                {
                    best_distances(i) = cost;
                }
            }
        }
        std::cout << "found new medoid" << new_medoid << std::endl;
        use_absolute = false; //use difference of loss for sigma and sampling, not absolute
    }
}

void KMediods::build_sigma(
    const arma::mat &data,
    arma::rowvec &best_distances,
    arma::rowvec &sigma,
    arma::uword batch_size,
    bool use_absolute)
{
    size_t N = data.n_cols;
    uvec tmp_refs = arma::randperm(N, batch_size); //without replacement, requires updated version of armadillo

    arma::rowvec sample(batch_size);
    // for each possible swap
    for (size_t i = 0; i < N; i++)
    {

        //gather a sample of points
        for (size_t j = 0; j < batch_size; j++)
        {
            double cost = arma::norm(data.col(i) - data.col(tmp_refs(j)));
            if (use_absolute) {
                sample(j) = cost;
            } else {
                sample(j) = cost < best_distances(tmp_refs(j)) ? cost : best_distances(tmp_refs(j));
                sample(j) -= best_distances(tmp_refs(j));
            }
            
        }
        sigma(i) = arma::stddev(sample);
    }
}

// switch this to a difference
// forcibly inline this in the future and directly write to estimates
arma::Row<double> KMediods::build_target(
    const arma::mat &data,
    arma::uvec &target,
    size_t batch_size,
    arma::Row<double> &best_distances,
    bool use_absolute)
{
    size_t N = data.n_cols;
    arma::Row<double> estimates(target.n_rows, arma::fill::zeros);
    //uvec tmp_refs = randi<uvec>(batch_size, distr_param(0, N - 1)); //with replacement
    uvec tmp_refs = arma::randperm(N, batch_size); //without replacement, requires updated version of armadillo
    double total = 0;
    for (size_t i = 0; i < target.n_rows; i++)
    {
        double total = 0;
        for (size_t j = 0; j < tmp_refs.n_rows; j++)
        {
            double cost = arma::norm(data.col(tmp_refs(j)) - data.col(target(i)), 2);
            if (use_absolute) {
                total += cost;
            } else {
                total += cost < best_distances(tmp_refs(j)) ? cost : best_distances(tmp_refs(j));
                total -= best_distances(tmp_refs(j));
            }
        }
        estimates(i) =total;
    }
    return estimates;
}



void KMediods::swap_sigma(
    const arma::mat &data,
    arma::mat &sigma,
    size_t batch_size,
    arma::Row<double> &best_distances,
    arma::Row<double> &second_best_distances,
    arma::urowvec &assignments)
{
    size_t N = data.n_cols;
    size_t K = sigma.n_rows;
    uvec tmp_refs = arma::randperm(N, batch_size); //without replacement, requires updated version of armadillo
    //uvec tmp_refs = arma::randperm(N, N); //without replacement, requires updated version of armadillo

    arma::vec sample(batch_size); // declare in outer loop, single allocation

    // for each considered swap
    for (size_t i = 0; i < K * N; i++)
    {
        // extract data point of swap
        size_t n = i / K;
        size_t k = i % K;

        // calculate change in loss for some subset of the data
        for (size_t j = 0; j < batch_size; j++)
        {
            double cost = arma::norm(data.col(n) - data.col(tmp_refs(j)), 2);

            if (k == assignments(tmp_refs(j)))
            {
                if (cost < second_best_distances(tmp_refs(j)))
                {
                    sample(j) += cost;
                }
                else
                {
                    sample(j) += second_best_distances(tmp_refs(j));
                }
            }
            else
            {
                if (cost < best_distances(tmp_refs(j)))
                {
                    sample(j) += cost;
                }
                else
                {
                    sample(j) += best_distances(tmp_refs(j));
                }
            }
            sample(j) -= best_distances(tmp_refs(j));
        }
        sigma(k, n) = arma::stddev(sample);
    }
}

arma::vec KMediods::swap_target(
    const arma::mat &data,
    arma::Row<size_t> &medoid_indices,
    arma::uvec &targets,
    size_t batch_size,
    arma::Row<double> &best_distances,
    arma::Row<double> &second_best_distances,
    arma::urowvec &assignments)
{
    size_t N = data.n_cols;
    arma::vec estimates(targets.n_rows, arma::fill::zeros);
    uvec tmp_refs = arma::randperm(N, batch_size); //without replacement, requires updated version of armadillo

    // for each considered swap
    for (size_t i = 0; i < targets.n_rows; i++)
    {
        double total = 0;
        // extract data point of swap
        size_t n = targets(i) / medoid_indices.n_cols;
        size_t k = targets(i) % medoid_indices.n_cols;
        // calculate total loss for some subset of the data
        for (size_t j = 0; j < batch_size; j++)
        {
            double cost = arma::norm(data.col(n) - data.col(tmp_refs(j)), 2);
            // the swap makes a better medoid
            if (k == assignments(tmp_refs(j)))
            {
                if (cost < second_best_distances(tmp_refs(j)))
                {
                    total += cost;
                }
                else
                {
                    total += second_best_distances(tmp_refs(j));
                }
            }
            else
            {
                if (cost < best_distances(tmp_refs(j)))
                {
                    total += cost;
                }
                else
                {
                    total += best_distances(tmp_refs(j));
                }
            }
            total -= best_distances(tmp_refs(j));
        }
        // total currently depends on the batch size, which seems distinctly wrong. maybe.
        uword temp = medoid_indices(k);
        medoid_indices(k) = n;
        //estimates(i) = calc_loss(data, medoid_indices.n_cols, medoid_indices);
        estimates(i) = total;
        medoid_indices(k) = temp;

        //estimates(i) = total;
        //std::cout << "estimate for n->k " << n << "->" << k << " :" << estimates(i) << " " << total << std::endl;
    }
    return estimates;
}

void calc_best_distances_swap(
    const arma::mat &data,
    const arma::mat &medoids,
    arma::Row<double> &best_distances,
    arma::Row<double> &second_distances,
    arma::urowvec &assignments)
{
    for (size_t i = 0; i < data.n_cols; i++)
    {
        double best = std::numeric_limits<double>::infinity();
        double second = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < medoids.n_cols; k++)
        {
            double cost = arma::norm(data.col(i) - medoids.col(k), 2);
            if (cost < best)
            {
                assignments(i) = k;
                second = best;
                best = cost;
            }
            else if (cost < second)
            {
                second = cost;
            }
        }
        best_distances(i) = best;
        second_distances(i) = second;
    }
}

void KMediods::swap(const arma::mat &data,
                    const size_t clusters,
                    arma::Row<size_t> &medoid_indicies,
                    arma::mat &medoids)
{

    size_t N = data.n_cols;
    size_t this_batch_size = 20;
    double p = (10);

    arma::mat sigma(clusters, N);

    arma::Row<double> best_distances(N);
    arma::Row<double> second_distances(N);
    arma::urowvec assignments(N);

    // does this need to be calculated in every iteration?
    size_t iter = 0;
    bool swap_performed = true;

    // initialize matrices -> should move declaration outside of loop
    arma::umat candidates(clusters, N, arma::fill::ones);
    arma::umat exact_mask(clusters, N, arma::fill::zeros);
    arma::mat estimates(clusters, N, arma::fill::zeros);
    arma::mat lcbs(clusters, N);
    arma::mat ucbs(clusters, N);
    arma::umat T_samples(clusters, N, arma::fill::zeros);

    // continue making swaps while loss is decreasing
    while (swap_performed && iter < maxIterations)
    {
        iter++;

        // calculate quantities needed for swap, best_distances and sigma
        calc_best_distances_swap(data, medoids, best_distances, second_distances, assignments);

        swap_sigma(data, sigma, this_batch_size, best_distances, second_distances, assignments);

        candidates.fill(1);
        exact_mask.fill(0);
        estimates.fill(0);
        T_samples.fill(0);

        size_t original_batch_size = 100;
        size_t step_count = 0;

        // while there is at least one candidate (double comparison issues)
        while (arma::accu(candidates) > 0.5)
        {
            calc_best_distances_swap(data, medoids, best_distances, second_distances, assignments);

            // compute exactly if it's been samples more than N times and hasn't been computed exactly already
            arma::umat compute_exactly = ((T_samples + this_batch_size) >= N) != (exact_mask);
            arma::uvec targets = arma::find(compute_exactly);
            //cout << "targets for compute exactly "<< targets << std::endl;

            if (targets.size() > 0)
            {
                size_t n = targets(0) / medoids.n_cols;
                size_t k = targets(0) % medoids.n_cols;
                std::cout << "n, k -> " << n << " " << k << std::endl;
                std::cout << "COMPUTING EXACTLY " << targets.size() << " out of " << candidates.size() << std::endl;
                arma::vec result = swap_target(data, medoid_indicies, targets, N, best_distances, second_distances, assignments);
                estimates.elem(targets) = result;
                ucbs.elem(targets) = result;
                lcbs.elem(targets) = result;
                exact_mask.elem(targets).fill(1);
                T_samples.elem(targets) += N;

                candidates = (lcbs < ucbs.min()) && (exact_mask == 0);

                // this is my own modification?
            }
            if (arma::accu(candidates) < 0.5)
            {
                std::cout << "yeeting away" << std::endl;
                break;
            }
            targets = arma::find(candidates);
            arma::vec result = swap_target(data, medoid_indicies, targets, this_batch_size, best_distances, second_distances, assignments);
            estimates.elem(targets) = ((T_samples.elem(targets) % estimates.elem(targets)) + (result * this_batch_size)) / (this_batch_size + T_samples.elem(targets));
            T_samples.elem(targets) += this_batch_size;
            //std::cout << "estimates:" << arma::mean(arma::mean(estimates)) << std::endl;
            arma::vec adjust(targets.n_rows);
            //std::cout << "result of log " << ::log(p) << std::endl;
            adjust.fill(::log(p));
            arma::vec cb_delta = sigma.elem(targets) % arma::sqrt(adjust / T_samples.elem(targets));
            //std::cout << "cb_delta:" << arma::mean(arma::mean(cb_delta)) << std::endl;

            ucbs.elem(targets) = estimates.elem(targets) + cb_delta;
            lcbs.elem(targets) = estimates.elem(targets) - cb_delta;
            //std::cout << "ucbs:" << arma::mean(arma::mean(ucbs)) << " lcbs:" << arma::mean(arma::mean(lcbs)) << std::endl;
            candidates = (lcbs < ucbs.min()) != (exact_mask);
            targets = arma::find(candidates);
            step_count++;
        }
        // now switch medoids
        uword new_medoid = lcbs.index_min();
        // extract medoid of swap
        size_t k = new_medoid % medoids.n_cols;

        // extract data point of swap
        size_t n = new_medoid / medoids.n_cols;
        swap_performed = medoid_indicies(k) != n;
        std::cout << (swap_performed ? ("swap performed") : ("no swap performed")) << std::endl;
        std::cout << "lcbs means is " << lcbs.min() << std::endl;
        medoid_indicies(k) = n;
        medoids.col(k) = data.col(medoid_indicies(k));
        std::cout << medoid_indicies << std::endl;
        calc_best_distances_swap(data, medoids, best_distances, second_distances, assignments);
        std::cout << "best distance sum:" << arma::accu(best_distances) << std::endl;
        std::cout << "calced distance:" << calc_loss(data, medoids.n_cols, medoid_indicies) << std::endl;
        std::cout << "medoids shape" << medoids.n_rows << " cols:" << medoids.n_cols << std::endl;
    }
    // done with swaps at this point
}

double KMediods::calc_loss(const arma::mat &data,
                           const size_t clusters,
                           arma::Row<size_t> &medoid_indices)
{
    double total = 0;

    for (size_t i = 0; i < data.n_cols; i++)
    {
        double cost = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < clusters; k++)
        {
            if (arma::norm(data.col(medoid_indices(k)) - data.col(i), 2) < cost)
            {
                cost = arma::norm(data.col(medoid_indices(k)) - data.col(i), 2);
            }
        }
        total += cost;
    }
    return total;
}