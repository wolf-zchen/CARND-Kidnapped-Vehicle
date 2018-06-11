/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  // Initialize random engine
	default_random_engine gen;

	// Number of best_particle
	num_particles = 100;

	// Create normal distributions for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Init particles
	for (int i = 0; i < num_particles; i++){
		Particle p;
		p.id = i;
		p.x =  dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(p.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	for (int i=0; i < num_particles; i++)
	{
		double new_x;
		double new_y;
		double new_theta;

		if (yaw_rate == 0)
		{
			new_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
			new_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
			new_theta = particles[i].theta;
		}
		else
		{
			new_x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			new_y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			new_theta = particles[i].theta + yaw_rate * delta_t;
		}

		// Create normal distributions for x, y and theta
		normal_distribution<double> dist_x(new_x, std_pos[0]);
		normal_distribution<double> dist_y(new_y, std_pos[1]);
		normal_distribution<double> dist_theta(new_theta, std_pos[2]);

		// Add noise
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	// Loop over all noisy_observations
	for (unsigned int i=0; i < observations.size(); i++)
	{
		// Grab current observation
		LandmarkObs obs = observations[i];

		// init minimum distance to maximum possible
    double min_dist = numeric_limits<double>::max();

    // init id of landmark from map placeholder to be associated with the observation
    int map_id = -1;

    for (unsigned int j = 0; j < predicted.size(); j++)
		{
      // grab current prediction
      LandmarkObs pred = predicted[j];

      // get distance between current/predicted landmarks
      double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);

      // find the predicted landmark nearest the current observed landmark
      if (cur_dist < min_dist)
			{
        min_dist = cur_dist;
        map_id = pred.id;
			}
		}

		// set the observation's id to the nearest predicted landmark's id
    observations[i].id = map_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Loop over each particle
	for (int i=0; i<num_particles; i++)
	{
		// Get x,y, the coordinates
		double px = particles[i].x;
		double py = particles[i].y;
		double ptheta = particles[i].theta;

		// Check range, get landmark locations within sensor range
		vector<LandmarkObs> predictions;

		// Loop over each landmark
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			// Get id and x,y coordinates
      double lm_x = map_landmarks.landmark_list[j].x_f;
      double lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;

			// If landmark is within sensor range
			double d = dist(lm_x, lm_y, px, py);
			if (d < sensor_range)
			{
				LandmarkObs new_obs;
				new_obs.id = lm_id;
				new_obs.x  = lm_x;
				new_obs.y  = lm_y;
				predictions.push_back(new_obs);
			}
		}

		// Transform from vehicle coords to map coordinates
		vector<LandmarkObs> trans_observations;
		for (unsigned int j = 0; j < observations.size(); j++)
		{
      LandmarkObs trans_obs;
			trans_obs.x = cos(ptheta)*observations[j].x - sin(ptheta)*observations[j].y + px;
      trans_obs.y = sin(ptheta)*observations[j].x + cos(ptheta)*observations[j].y + py;
			trans_obs.id = observations[j].id;
      trans_observations.push_back(trans_obs);
		}

		// Perform data association
		dataAssociation(predictions, trans_observations);

		// Reset weight
		particles[i].weight = 1.0;

		for (unsigned int j=0; j < trans_observations.size(); j++)
		{
			// Observation coords
			double ob_x, ob_y, pred_x, pred_y;
			ob_x = trans_observations[j].x;
			ob_y = trans_observations[j].y;
			int associated_pred = trans_observations[j].id;

			for (unsigned int k = 0; k < predictions.size(); k++)
			{
				if (predictions[k].id == associated_pred)
				{
					pred_x = predictions[k].x;
					pred_y = predictions[k].y;
				}
			}

			// calculate weight for this observation with multivariate Gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = ( 1/(2*M_PI*s_x*s_y)) * \
			                exp( -( pow(pred_x - ob_x,2)/(2*pow(s_x, 2))\
											+ (pow(pred_y - ob_y,2)/(2*pow(s_y, 2))) ) );

      // product of this obersvation weight with total observations weight
      particles[i].weight *= obs_w;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	//temp vectors for the new set of resampled particles
  vector<Particle> new_particles;

	// get all of the current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // generate random starting index for resampling wheel
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
	default_random_engine gen;
  auto index = uniintdist(gen);

  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // uniform random distribution [0.0, max_weight)
  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  // spin the resample wheel!
  for (int i = 0; i < num_particles; i++)
	{
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index])
		{
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
