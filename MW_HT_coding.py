'''
Michael Wiest 09/21/15

This script performs three main functions:
1: It ranks hotels based on the output of a supposedly true model.score function.
In my case, the score function is very simplistic and is a placeholder for an
actual model.

2: It returns an average rank for a hotel at a price. The average rank is based on weighting
results of model.rank by the distribution of users that have called model.rank. This function
can handle minimum category percents and preferred categories. In the case of preferred categories
the function populates the remaining space in the output array with hotels in the preferred category.

3: It returns a price necessary to achieve a given rank on average. This is a two stage algorithm that
makes a first prediction of price based on a linear relationship between price and rank and next
performs something like binary search to find the proper price. See more about this function above it.

USAGE (using the files in the zip'd directory I sent. Different, but similarly structured inputs could
    also be used):
$ python MW_HT_coding.py big_hotel_list.txt User_list.txt prices.txt

The script assigns the prices in prices.txt to a subset of the hotels read in from
big_hotel_list.

OUTPUT: The test function calls a few instances of the above functions and prints a few of the results.
The output will be different each time because the relationship between user and hotel is randomly 
generated each time the script is run.

COMPLEXITY:
-rank: R = O(Nlog(N) + C*N) Where N is length of hotels and C is length of categories
 with minimum percents. Nlog(N) because of sorting the hotel array.
-price_to_rank: PtR = O(Nlog(N) + M*R) where M is length of users in known user base and R is 
the complexity of model.rank
-rank_to_price: RtP = O(PtR * I) where I is the iterations required to converge. In theory I 
should be close to log(N).
'''

import sys
import numpy as np
import time
import copy

'''
Model object that performs all of the functions specified in the prompt. user_hotel_scores 
and hotel_indices and user_indices are for accessing the dummy data generated for scoring.
Also has a count of how many times a specific category was specified as preferred.
'''
class Model:
    def __init__(self, hotels, users, user_hotel_scores, hotel_indices, user_indices):
        self.hotels = hotels
        self.user_hotel_scores = user_hotel_scores
        self.hotel_indices = hotel_indices
        self.user_indices = user_indices
        self.score_scale = 10000
        self.users_called = {}
        self.preferred_categories = {}
        self.rank_calls = 0
        self.get_categories()

    '''
    Return a score based on price and the user and hotel ceofficients.
    A higher score is considered to be more attractive. Scores are scaled 
    by a score_scale factor for readability.
    '''
    def score(self, user, hotel, price):
        return (self.user_hotel_scores[self.hotel_indices[hotel], self.user_indices[user]] / price) * self.score_scale

    def get_categories(self):
        self.categories = []
        for i in xrange(len(self.hotels)):
            if not self.hotels[i].category in self.categories:
                self.categories.append(self.hotels[i].category)
      
    def get_hotel_by_id(self, id):
        for i in xrange(len(self.hotels)):
            if self.hotels[i].id == id:
                return self.hotels[i]
   
    '''
    Return a list of hotel ids corresponding to a best to worst ranking of hotels based on the
    score from model.score. The count_calls option is for deciding whether or not to count a 
    given call to rank as contributing to the distribution of users calling rank. If any of
    the specified criteria for preferred category or min_category_representation cannot be met
    this returns False. If n > length of hotels in question then an array of the length of hotels
    is returned.
    '''
    def rank(self, user, hotels, prices, n, reverse=False, min_category_representation=None,
             preferred_category=None, return_hotels=False, count_calls=True):
        if preferred_category and preferred_category not in self.categories:
            return False
        if user not in self.user_indices:
            return False
        # So that calculations of minimum percentage won't be wrong.
        if n > len(hotels):
            n = len(hotels)

        if count_calls:
            self.rank_calls += 1
            if user in self.users_called:
                self.users_called[user] += 1
            else:
                self.users_called[user] = 1

        rank_scores = np.array([0.0] * len(hotels))
        rank_hotels = np.array([None] * len(hotels))
        category_counts = {}

        for i in xrange(len(hotels)):
            score = self.score(user, hotels[i].id, prices[i])
            rank_scores[i] = score
            rank_hotels[i] = hotels[i]

            if hotels[i].category in category_counts:
                category_counts[hotels[i].category] += 1
            else:
                category_counts[hotels[i].category] = 1
        
        if not reverse:
            rank_hotels = rank_hotels[np.argsort(rank_scores*-1)]
            rank_scores = rank_scores[np.argsort(rank_scores*-1)]
        else:
            rank_hotels = rank_hotels[np.argsort(rank_scores)]
            rank_scores = rank_scores[np.argsort(rank_scores)]
        
        output_hotels = []
        output_scores = []
        
        '''
        Fill the output array up with required categories to satisfy min_category_representation.
        '''
        if min_category_representation:
            for category, percent in min_category_representation.iteritems():
                # Check if criteria can be met.
                try:
                    if float(category_counts[category]) / n < percent:
                        return False
                except:
                    return False
                hotel_temp, score_temp = self.hotels_by_category(rank_hotels, rank_scores, category, percent, n)
                output_hotels += hotel_temp
                output_scores += score_temp
        
        '''
        If there is a preferred category fill the output array with as many hotels in that category
        as possible. Another approach would be stop filling with the preferred category if the score is
        below a specified threshold.
        '''
        if preferred_category:
            if preferred_category not in self.preferred_categories:
                self.preferred_categories[preferred_category] = 1
            else:
                self.preferred_categories[preferred_category] += 1
            for i in xrange(len(rank_hotels)):
                if rank_hotels[i] not in output_hotels and rank_hotels[i].category == preferred_category and len(output_hotels) < n:
                    output_hotels.append(rank_hotels[i])
                    output_scores.append(rank_scores[i])

        '''
        If there is any remaining space in the output fill it with the next highest ranked hotels.
        '''
        for i in xrange(len(rank_hotels)):
            if rank_hotels[i] not in output_hotels and len(output_hotels) < n:
                output_hotels.append(rank_hotels[i])
                output_scores.append(rank_scores[i])
        
        output_hotels = np.array(output_hotels)
        output_scores = np.array(output_scores)
        if not reverse:
            output_hotels = output_hotels[np.argsort(output_scores * -1)]
            output_scores = output_scores[np.argsort(output_scores * -1)]
        else:
            output_hotels = output_hotels[np.argsort(output_scores)]
            output_scores = output_scores[np.argsort(output_scores)]

        if not return_hotels:
            return [x.id for x in output_hotels]
        else:
            return output_hotels
    
    '''
    This is a helper function for the rank function that given a list of ordered hotels
    returns a list of hotels and scores that satisfy the minimum percentage criteria. 
    '''
    def hotels_by_category(self, rank_hotels, rank_scores, category, min_perc, n):
        output_hotels = []
        output_scores = []
        for i in xrange(len(rank_hotels)):
            if rank_hotels[i].category == category:
                output_hotels.append(rank_hotels[i])
                output_scores.append(rank_scores[i])
            if float(len(output_hotels)) / n >= min_perc and len(output_hotels) <= n:
                return output_hotels, output_scores

    '''
    This returns the average rank of a hotel at a given price among all other
    specified hotels and prices. This does not compare the argument hotel against
    itself. The way for accounting for all users is to weight the rank of a given
    set of hotel ranks by the percentage of time that that particular user has 
    called the rank function.
    This optionally can return the relative position of this hotel among all averages.
    '''
    def price_to_rank(self, hotels, prices, hotel_id_to_predict, new_price,
                      relative_position=False, full_list=False):
        argument_hotel = self.get_hotel_by_id(hotel_id_to_predict)
        if argument_hotel not in self.hotels:
            return False
        try:
            index = hotels.index(argument_hotel)
            del hotels[index]
            del prices[index]
        except:
            pass

        hotels.append(argument_hotel)
        prices.append(new_price)

        hotel_ranks = dict(zip(hotels, [0]*len(hotels)))

        for user, count in self.users_called.iteritems():
            scaling_factor = float(count) / self.rank_calls

            ranked_hotels = self.rank(user, hotels, prices, len(hotels),
                            return_hotels=True, count_calls=False)

            for i in xrange(len(ranked_hotels)):
                rank = i + 1
                hotel_ranks[ranked_hotels[i]] += rank * scaling_factor

        output_hotels = []
        output_ranks = []
        for hotel, rank in hotel_ranks.iteritems():
            output_hotels.append(hotel)
            output_ranks.append(rank)
        
        output_hotels = np.array(output_hotels)
        output_ranks = np.array(output_ranks)
        output_hotels = output_hotels[np.argsort(output_ranks)]
        output_ranks = output_ranks[np.argsort(output_ranks)]
        
        if full_list:
            return output_ranks, output_hotels

        index = output_hotels.tolist().index(argument_hotel)
        if relative_position:
            return index + 1
        else:
            return output_ranks[index]
 

    '''
    This function returns a predicted price necessary for a hotel to achieve new_rank amongst
    all hotels and prices in the argument. This algorithm has two main steps:
        -It first finds the hotel closest to new_rank that is on the other side of new_rank
        (ie, if hotel in question is of rank 6 and new_rank is 3, then the target point is 
        the hotel with the largest rank less than 3). It then assumes a linear realationship
        between rank and price and predicts the price necessary to achieve new_rank.
        -Second the algorithm uses what is essentially binary search between two changing
        fences to zero in on the price necessary for the given rank (assuming error with
        accuracy).
    This could potentially return a higher price and still fall within the appropriate accuracy
    window if a price was considered converged to found_rank only when: 
        (1). found_rank > new_rank and found_rank - accuracy < new_rank
    My algorithm considers a match if:
        (2). found_rank + accuracy > new_rank or found_rank - accuracy < new_rank
    So in theory my algorithm may report too low of a price. This can be easily remedied by
    following criteria found in (1) instead of (2).

    A way to make this faster is when calling price_to_rank to have an option to only recalculate
    the rank for a specified point that way calculations aren't being redone every time for points
    that aren't changing.

    A major drawback with this method is that if accuracy is too small, typically ~0.1, then this
    function won't converge and will isntead oscillate around new_rank. I could solve this in the
    future by evolving step size as the values approach new_rank.
    '''
    def rank_to_price(self, hotels, prices, hotel_id_to_predict, new_rank, accuracy=0.2):
        start = time.time()
        if new_rank > len(hotels):
            return False
        
        min_guess = 0
        max_guess = max(prices)

        argument_hotel = self.get_hotel_by_id(hotel_id_to_predict)
        if argument_hotel not in self.hotels:
            return False
        
        starting_price = prices[hotels.index(argument_hotel)]
        starting_ranks, starting_hotels = self.price_to_rank(copy.copy(hotels), copy.copy(prices),
                                            hotel_id_to_predict, starting_price, full_list=True)    
        starting_rank = starting_ranks[starting_hotels.tolist().index(argument_hotel)]
        
        '''
        This is for finding a target point with which to make a linear assumption and 
        guess about the next point.
        '''
        if starting_rank > new_rank:
            max_guess = starting_price
            try:
                target = max(starting_ranks[starting_ranks < new_rank])
                position = starting_ranks.tolist().index(target)
                new_price = prices[hotels.index(starting_hotels[position])]
                compare_rank = starting_ranks[position]
            except:
                new_price = prices[hotels.index(starting_hotels[0])]
                compare_rank = 0
        else:
            min_guess = starting_price
            try:
                target = min(starting_ranks[starting_ranks > new_rank])
                position = starting_ranks.tolist().index(target)
                new_price = prices[hotels.index(starting_hotels[position])]
                compare_rank = starting_ranks[position]
            except:
                new_price = prices[hotels.index(starting_hotels[len(hotels)])]
                compare_rank = len(hotels)
        try:
            slope = float((starting_rank - compare_rank)) / (starting_price - new_price)
        except:
            # If the slope is undefined. Make it very large instead.
            slope = 1e10

        y_int = starting_rank - slope * starting_price
        projected_price = float((new_rank - y_int)) / slope
        updated_rank = starting_rank

        '''
        This is the binary search section.
        '''
        while not (updated_rank - accuracy) < new_rank or not (updated_rank + accuracy) > new_rank:
            prices[hotels.index(argument_hotel)] = projected_price
            
            ranks_new, hotels_new = self.price_to_rank(copy.copy(hotels), copy.copy(prices),
                                    hotel_id_to_predict, projected_price, full_list=True)
            updated_rank = ranks_new[hotels_new.tolist().index(argument_hotel)]

            if updated_rank > new_rank:
                max_guess = projected_price
                # If the function gets stuck.
                if max_guess - min_guess < 1e-5:
                    min_guess = float(min_guess) / 3
    
                projected_price = float((min_guess + projected_price)) / 2                
                
            else:
                min_guess = projected_price
                # If the function gets stuck.
                if max_guess - min_guess < 1e-5:
                    max_guess = float(max_guess) * 3
                projected_price = float((max_guess + projected_price)) / 2            

            if time.time() - start > 5:
                return False
        return projected_price

'''
Hotel object. In this case the name and id are the same. But in actual use the id would likely be a 
unique alphanumeric sequence so the name could be repeated.            
'''
class Hotel:
    def __init__(self, category, name, id):
        self.category = category
        self.name = name
        self.id = id

def main():
    hotel_list = np.loadtxt(sys.argv[1], dtype='string', delimiter=',')
    user_list = np.loadtxt(sys.argv[2], dtype='string')
    hotel_prices = np.loadtxt(sys.argv[3], dtype='float').tolist()

    hotels = generate_hotels(hotel_list)
    scores, hotel_indices, user_indices = generate_scores(hotel_list[:, 0], user_list)
    model = Model(hotels, user_list, scores, hotel_indices, user_indices)

    test(model, hotel_prices)
   

'''    
Generate a matrix of random values 0 to 1 that represent a score for a given user and given hotel.
This is used for the score function.
'''
def generate_scores(hotel_list, user_list):
    hotel_indices = dict(zip(hotel_list, xrange(len(hotel_list))))
    user_indices = dict(zip(user_list, xrange(len(user_list))))
    scores = np.random.rand(len(hotel_list), len(user_list))
    return scores, hotel_indices, user_indices

'''
Build out all of the hotel objects from the input files.
'''
def generate_hotels(hotel_list):
    hotels = []
    for i in xrange(hotel_list.shape[0]):
        hotel = Hotel(hotel_list[i, 1], hotel_list[i, 0], hotel_list[i, 0])
        hotels.append(hotel)
    return hotels


def test(model, hotel_prices):
    test_hotels = []
    # Get a subset of hotels for ranking.
    for i in xrange(len(model.hotels)):
        test_hotels.append(model.hotels[i])
        if len(test_hotels) == len(hotel_prices):
            break

    category = {'CHARMING': 0.25, 'LUXE': 0.1}
    model.rank('User_B', test_hotels, hotel_prices, 3,
        min_category_representation=category, preferred_category='HIP')
    model.rank('User_A', test_hotels, hotel_prices, 8,
        min_category_representation=category, preferred_category='LUXE')
    model.rank('User_G', test_hotels, hotel_prices, 5,
        min_category_representation=category, preferred_category='HIP')
    # This user isn't in the set.
    model.rank('User_0', test_hotels, hotel_prices, 11,
        min_category_representation=category, preferred_category='HIP')
    model.rank('User_Z', test_hotels, hotel_prices, 1,
        min_category_representation=category, preferred_category='SOLID')
    model.rank('User_T', test_hotels, hotel_prices, 2,
        min_category_representation=category, preferred_category='BASIC')
    model.rank('User_B', test_hotels, hotel_prices, 6,
        min_category_representation=category, preferred_category='HIP')
    model.rank('User_C', test_hotels, hotel_prices, 3,
        min_category_representation=category, preferred_category='HIP')
    
    category = {'CHARMING': 0.1}
    # Test that n will reduce to length of hotels.
    print model.rank('User_C', test_hotels, hotel_prices, 100, min_category_representation=category)
    print model.rank('User_C', test_hotels, hotel_prices, 1, min_category_representation=category)
    
    # This hotel isn't in the set.
    model.price_to_rank(test_hotels, hotel_prices, 'Hotel_D', 40, relative_position=True)
    print model.price_to_rank(test_hotels, hotel_prices, 'Hotel_100', 50)
    print model.rank_to_price(test_hotels, hotel_prices, 'Hotel_8', 3)
    # This will time out.
    print model.rank_to_price(test_hotels, hotel_prices, 'Hotel_9', 0.1)


if __name__ == '__main__':
    main()
