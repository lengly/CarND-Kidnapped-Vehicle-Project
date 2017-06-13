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
	double std_x = std[0];
	double std_y = std[1];
	double std_psi = std[2];

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_psi(theta, std_psi);

	default_random_engine gen;

	num_particles = 100;
	weights.clear();
	particles.clear();
	for(int i = 0; i < num_particles; i++) {
		weights.push_back(1);

		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_psi(gen);
		p.weight = 1;

		particles.push_back(p);
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_psi = std_pos[2];
	for(int i = 0; i < particles.size(); i++) {
		double x = particles[i].x + velocity / yaw_rate 
			* (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
		double y = particles[i].y + velocity / yaw_rate
			* (cos(particles[i].theta - cos(particles[i].theta + yaw_rate * delta_t)));
		double psi = particles[i].theta + yaw_rate * delta_t;

		normal_distribution<double> dist_x(x, std_x);
		normal_distribution<double> dist_y(y, std_y);
		normal_distribution<double> dist_psi(psi, std_psi);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_psi(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {
		double mindis = -1;
		for (int j = 0; j < predicted.size(); j++) {
			double dis = dist(observations[i].x, observations[i].y, predicted[i].x, predicted[i].y);
			if (mindis == -1 || mindis > dis) {
				mindis = dis;
				observations[i].id = predicted[j].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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
	double sigma_x, sigma_y, sigma_x_2, sigma_y_2, sigma_xy;
	sigma_x = std_landmark[0];
    sigma_y = std_landmark[1];
    sigma_x_2 = 2 * pow(sigma_x, 2);
    sigma_y_2 = 2 * pow(sigma_y, 2);
    sigma_xy = 2*M_PI*sigma_x*sigma_y;

	for (int i = 0; i < particles.size(); i++) {

		std::vector<LandmarkObs> predicted;
		for(Map::single_landmark_s landmark : map_landmarks.landmark_list) {
			if (dist(landmark.x_f, landmark.y_f, particles[i].x, particles[i].y) <= sensor_range) {
				predicted.push_back(LandmarkObs({landmark.id_i, landmark.x_f, landmark.y_f}));
			}
		}

		std::vector<LandmarkObs> trans_obs;
		for (LandmarkObs obs : observations) {
			double x, y;
			x = particles[i].x + particles[i].x * cos(particles[i].x) - particles[i].y * sin(particles[i].theta);
			y = particles[i].y + particles[i].x * sin(particles[i].x) + particles[i].y * sin(particles[i].theta);
			trans_obs.push_back(LandmarkObs({0, x, y}));
		}

		dataAssociation(predicted, trans_obs);

		double total_weight = 1;
		for(int j = 0; j < trans_obs.size(); j++) {
			Map::single_landmark_s near = map_landmarks.landmark_list[trans_obs[j].id];
			total_weight *= 1 / sigma_xy * exp(-(pow(trans_obs[i].x - near.x_f, 2)/sigma_x_2 + pow(trans_obs[i].y - near.y_f, 2)/sigma_y_2));
		}
		particles[i].weight = total_weight;
		weights[i] = total_weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> dist_w(weights.begin(), weights.end());
	vector<Particle> resample_particles;
	for(int i = 0; i < particles.size(); i++) {
		resample_particles.push_back(particles[dist_w(gen)]);
	}
	particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
