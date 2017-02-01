import unittest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal

from balsa.cheval import LinkedDataFrame


class TestA(unittest.TestCase):

    def testGetItem(self):
        sa = pd.Series([1, 2, 3])
        df = LinkedDataFrame({'a': sa})

        result = df['a']

        assert_series_equal(sa, result, check_names=False)

    def testGetAttr(self):
        sa = pd.Series([1, 2, 3])
        df = LinkedDataFrame({'ldf_test_column': sa})

        result = df.ldf_test_column

        assert_series_equal(sa, result, check_names=False)


def generate_counter(reps):
    new_array = np.zeros(reps.sum())

    i = 0
    for rep in reps:
        val = 0
        for j in range(rep):
            new_array[i] = val
            val += 1
            i += 1
    return new_array


class TestLookups(unittest.TestCase):

    # TODO: Write test for __dir__
    # TODO: Write test for refresh_links
    # TODO: Write test for __getitem__
    # TODO: Write test for remove_link

    def setUp(self):
        randomizer = np.random.RandomState(12345)

        n_zones = 3
        n_hh = 10
        max_hh_size = 4
        trips_probability = pd.Series({1: 0.1, 2: 0.7, 3: 0.2})

        zone_index = np.arange(0, n_zones)
        zone_area = randomizer.uniform(10.0, 100.0, size=n_zones)
        zones = pd.DataFrame({'area': zone_area},
                             index=pd.Index(zone_index, name='zone_id'))

        hh_index = np.arange(0, n_hh)
        hh_dwellings = randomizer.choice(['house', 'apartment'], size=n_hh, replace=True, p=[0.3, 0.7])
        hh_zone = randomizer.choice(zone_index, size=n_hh, replace=True)
        hh_vehicles = randomizer.choice([0, 1, 2, 3], size=n_hh, replace=True)
        households = LinkedDataFrame({'dwelling_type': hh_dwellings, 'zone_id': hh_zone, 'vehicles': hh_vehicles},
                                     index=pd.Index(hh_index, name='hhid'))

        hh_repetitions = randomizer.randint(1, max_hh_size, n_hh)
        person_hh = np.repeat(hh_index, hh_repetitions)
        person_index = generate_counter(hh_repetitions)
        person_ages = randomizer.randint(14, 50, len(person_index))
        person_sex = randomizer.choice(['M', 'F'], size=len(person_index), replace=True)
        persons = LinkedDataFrame({'age': person_ages, 'sex': person_sex},
                                  index=pd.MultiIndex.from_arrays([person_hh, person_index], names=['hhid', 'pid']))

        person_repetitions = randomizer.choice(trips_probability.index.values, size=len(persons), replace=True,
                                               p=trips_probability.values)
        trip_hh = np.repeat(persons.index.get_level_values(0).values, person_repetitions)
        trip_persons = np.repeat(persons.index.get_level_values(1).values, person_repetitions)
        trip_index = generate_counter(person_repetitions)
        trips = LinkedDataFrame(index=pd.MultiIndex.from_arrays([trip_hh, trip_persons, trip_index], names=['hhid', 'pid',
                                                                                                         'trip_id']))
        trips['km'] = randomizer.uniform(2.0, 50.0, size=len(trips))

        households.link_to(zones, 'zone', on_self='zone_id')
        households.link_to(persons, 'persons', levels='hhid')

        persons.link_to(households, 'household', levels='hhid')
        persons.link_to(trips, 'trips', levels=['hhid', 'pid'])

        trips.link_to(persons, 'person', levels=['hhid', 'pid'])
        trips.link_to(households, 'household', levels='hhid')

        self.zones = zones
        self.hh = households
        self.persons = persons
        self.trips = trips

    def tearDown(self):
        del self.zones, self.trips, self.hh, self.persons

    def testStandardLookup(self):
        oracle = pd.DataFrame(self.hh.to_dict()).zone_id
        result = self.hh.zone_id

        assert_series_equal(oracle, result, check_names=False, check_dtype=False)

    def testSingleLookup(self):
        oracle = self.hh.dwelling_type
        oracle = oracle.reindex(self.persons.index.get_level_values('hhid'))
        oracle.index = self.persons.index

        result = self.persons.household.dwelling_type

        assert_series_equal(oracle, result, check_names=False)

    def testDoubleLookup(self):
        result = self.persons.household.zone.area

        oracle = self.zones.area.reindex(self.hh.zone_id)
        oracle.index = self.hh.index
        oracle = oracle.reindex(self.persons.index.get_level_values('hhid'))
        oracle.index = self.persons.index

        assert_series_equal(oracle, result, check_names=False)

    def testAggregateCount(self):
        oracle = self.persons.age.groupby(level=['hhid']).count()

        result = self.hh.persons.count()

        assert_series_equal(oracle, result, check_names=False, check_dtype=False)

    def testAggregateCountWithExpression(self):
        oracle = self.persons.age.loc[self.persons.age > 20].groupby(
            level=['hhid']
        ).count().reindex(self.hh.index, fill_value=0)

        result = self.hh.persons.count("age > 20")

        assert_series_equal(oracle, result, check_names=False)

    def testMultiIndexLookup(self):
        oracle = self.persons.sex
        oracle = oracle.reindex(self.trips.index.droplevel('trip_id'))
        oracle.index = self.trips.index

        result = self.trips.person.sex

        assert_series_equal(oracle, result, check_names=False)

    def testSliceSubset(self):
        persons_raw = pd.DataFrame(self.persons) # Make a copy as a plain ol' DataFrame
        person_subset = persons_raw.sample(5, replace=False, random_state=12345)

        oracle = self.hh.dwelling_type
        oracle = oracle.reindex(person_subset.index.get_level_values('hhid'))
        oracle.index = person_subset.index

        person_subset_2 = self.persons.sample(5, replace=False, random_state=12345)
        assert person_subset.index.equals(person_subset_2.index)

        result = person_subset_2.household.dwelling_type

        assert_series_equal(oracle, result, check_names=False)

    def testSliceSuperset(self):
        persons_raw = pd.DataFrame(self.persons) # Make a copy as a plain ol' DataFrame
        person_superset = persons_raw.sample(100, replace=True, random_state=12345)

        oracle = self.hh.dwelling_type
        oracle = oracle.reindex(person_superset.index.get_level_values('hhid'))
        oracle.index = person_superset.index

        person_superset_2 = self.persons.sample(100, replace=True, random_state=12345)
        assert person_superset.index.equals(person_superset_2.index)

        result = person_superset_2.household.dwelling_type

        assert_series_equal(oracle, result, check_names=False)

if __name__ == '__main__':
    unittest.main()
