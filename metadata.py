import datetime
import csv
import numpy as np

nan = float('nan')

metadata_raw = [
[65555, int, 1327259329, 1327356134, 1327309855.5137975, 28237.952239130042, 0],
[1, str, None, None, None, None, None],
[64885, datetime.datetime, None, None, None, None, None],
[16800, float, -178.92500000000001, 179.21100000000001, -159.14119836778281, 26.74767524011239, 0],
[8342, float, 55.036999999999999, 71.626000000000005, 66.53757922355274, 3.4426639342989835, 0],
[1, str, None, None, None, None, None],
[219, float, 0.0, 3314.0, 6.77235908778888, 51.769138071783665, 0],
[29, float, -140.0, -109.0, -130.74352833498588, 3.1520512971719423, 0],
[26209, float, 401.6395579, 401.67991060000003, 401.65787084285711, 0.014282382007899122, 0],
[38, int, 0, 68, 42.067134467241246, 26.620460434692145, 0],
[8776, float, 46.021000000000001, 89.882999999999996, 66.583555731828241, 3.6088247789653005, 0],
[31336, float, -7.992, 89.974000000000004, 68.337729555335201, 12.003869177153453, 0],
[7, int, None, None, None, None, None],
[17225, float, -179.98699999999999, 179.929, -158.12360744413093, 31.182078273545585, 0],
[56757, float, -179.995, 180.0, -21.033411440774923, 130.64698019622699, 0],
[15, int, 1, 15, 4.9746472427732442, 2.4039345107845147, 0],
[5, int, 0, 4, 0.0033559606437342689, 0.069786351302635261, 0],
[5, int, 0, 4, 2.6839752879261689, 0.97545122273792406, 0],
[728, float, 0.0, 999.0, 341.04696819464573, 169.37470497525266, 0],
[174, int, 0, 255, 142.03281214247579, 82.297515194018942, 0],
[255, int, 0, 255, 155.06400732209596, 85.04541469161218, 0],
[251, int, 0, 255, 65.396430478224389, 79.171224147694574, 0],
[257, str, None, None, None, None, None],
[3, str, None, None, None, None, None],
[1, str, None, None, None, None, None],
[1, str, None, None, None, None, None],
[1, str, None, None, None, None, None],
[1, str, None, None, None, None, None],
[78, int, 2521, 41691, 21281.058592021967, 14214.950019017622, 0],
[110, str, None, None, None, None, None],
[1, str, None, None, None, None, None]
]

metadata_env = [
[65555, int, 1327259329, 1327356134, 1327309855.5137975, 28237.952239130042, 0],
[64885, datetime.datetime, None, None, None, None, None],
[16800, float, -178.92500000000001, 179.21100000000001, -159.14119836778281, 26.74767524011239, 0],
[8342, float, 55.036999999999999, 71.626000000000005, 66.53757922355274, 3.4426639342989835, 0],
[110, str, None, None, None, None, None],
[6017, float, -0.002839196, 0.001169741, -0.00020923727242201201, 0.00025372418999361626, 0],
[65547, float, 241.40554729999999, 292.62299890000003, 273.64484250390967, 8.3539664894566119, 0],
[45189, float, 0.0, 42333.0, 7521.4559724883902, 8667.6764867965921, 0],
[65306, float, 4.4473302139999999, 20023.396130000001, 12498.842840947857, 7303.4622922395283, 0],
[51157, float, 0.010075927, 0.065644897999999993, 0.017614277201510181, 0.002483761423055314, 0],
[65551, float, -21.080586709999999, 20.16687464, -1.0247233927469757, 4.8647953166610574, 0],
[65551, float, -20.810700090000001, 20.24135953, -1.45063101210373, 5.2434509217226912, 0],
[60327, float, 9.9999999999999998e-13, 1.0, 0.743102277597807, 0.29391160015624507, 0],
[65513, float, 94959.482359999995, 104380.62820000001, 100349.69809264252, 1158.5317688474115, 0],
[65551, float, 0.70862954, 39.72950247, 11.194576972507939, 6.4196639083671263, 0],
[65543, float, 241.0419503, 290.97467289999997, 273.1702382199237, 7.5404664544593087, 0],
[59658, float, -34.62561917, 132.37799050000001, 4.6072516428248953, 7.9962651401687506, 0],
[16, int, 60, 230, 203.13950118221342, 20.229768280919515, 0],
[4350, float, 0.0, 0.0057368339999999997, 8.0336033771768732e-05, 0.00022090229077901508, 0],
[44104, float, 247.408705, 273.16082, 270.4825847814156, 3.3151211856370324, 0],
[65535, float, 248.04584869999999, 291.0741865, 273.84102653578526, 6.9635705954699905, 0],
[28228, int, 0.0, 1.0, 0.64421309954882156, 0.41041479238520817, 0],
[63902, float, -9.2599999999999993e-22, 0.36346331700000001, 0.09168344566862939, 0.069889765193965211, 0]
]

metadata_human = [
[65555, int, 1327259329, 1327356134, 1327309855.5137975, 28237.952239130042, 0],
[1, str, None, None, None, None, None],
[64885, datetime.datetime, None, None, None, None, None],
[16800, float, -178.92500000000001, 179.21100000000001, -159.14119836778281, 26.74767524011239, 0],
[8342, float, 55.036999999999999, 71.626000000000005, 66.53757922355274, 3.4426639342989835, 0],
[1, str, None, None, None, None, None],
[219, float, 0.0, 3314.0, 6.77235908778888, 51.769138071783665, 0],
[29, float, -140.0, -109.0, -130.74352833498588, 3.1520512971719423, 0],
[26209, float, 401.6395579, 401.67991060000003, 401.65787084285711, 0.014282382007899122, 0],
[38, int, 0, 68, 42.067134467241246, 26.620460434692145, 0],
[8776, float, 46.021000000000001, 89.882999999999996, 66.583555731828241, 3.6088247789653005, 0],
[31336, float, -7.992, 89.974000000000004, 68.337729555335201, 12.003869177153453, 0],
[7, int, None, None, None, None, None],
[17225, float, -179.98699999999999, 179.929, -158.12360744413093, 31.182078273545585, 0],
[56757, float, -179.995, 180.0, -21.033411440774923, 130.64698019622699, 0],
[15, int, 1, 15, 4.9746472427732442, 2.4039345107845147, 0],
[5, int, 0, 4, 0.0033559606437342689, 0.069786351302635261, 0],
[5, int, 0, 4, 2.6839752879261689, 0.97545122273792406, 0],
[728, float, 0.0, 999.0, 341.04696819464573, 169.37470497525266, 0],
[174, int, 0, 255, 142.03281214247579, 82.297515194018942, 0],
[255, int, 0, 255, 155.06400732209596, 85.04541469161218, 0],
[251, int, 0, 255, 65.396430478224389, 79.171224147694574, 0],
[257, str, None, None, None, None, None],
[3, str, None, None, None, None, None],
[1, str, None, None, None, None, None],
[1, str, None, None, None, None, None],
[1, str, None, None, None, None, None],
[1, str, None, None, None, None, None],
[78, int, 2521, 41691, 21281.058592021967, 14214.950019017622, 0],
[110, str, None, None, None, None, None],
[1, str, None, None, None, None, None],
[65551, float, -0.0028391962584294354, 0.0011697411720044998, -0.000209237520917059, 0.00025372416250069783, 0],
[65551, float, 241.40554732619592, 292.62299888037978, 273.64484250372158, 8.3539664895058277, 0],
[45202, float, 0.0, 42333.0, 7521.455972485649, 8667.676486799639, 0],
[65392, float, 4.4473302142708393, 20023.396130280671, 12498.842840940844, 7303.4622922318822, 0],
[65359, float, 0.01007592656616658, 0.065644898359981191, 0.017614277200462339, 0.0024837614243530585, 0],
[-1, float, nan, nan, nan, nan, 56899],
[-1, float, nan, nan, nan, nan, 55580],
[65551, float, -21.080586709806965, 20.166874639963108, -1.0247233927550103, 4.8647953166599214, 0],
[65551, float, -20.810700088581143, 20.241359533719663, -1.4506310120959884, 5.2434509217153407, 0],
[60391, float, 9.9997701091814051e-13, 1.0, 0.74310227763850267, 0.2939116000553661, 0],
[-1, float, nan, nan, nan, nan, 57270],
[65551, float, 94959.482357808563, 104380.62816419511, 100349.69809270573, 1158.5317688390626, 0],
[65551, float, 0.70862954030334013, 39.729502468668741, 11.194576972510314, 6.4196639083603051, 0]
]

metadata_aggregated = [
[65555, int, 1327259329, 1327356134, 1327309855.5137975, 28237.952239130042, 0],                               # event-id
[1, str, None, None, None, None, None],                                                                        # visible
[64885, datetime.datetime, None, None, None, None, None],                                                      # timestamp
[16800, float, -178.92500000000001, 179.21100000000001, -159.14119836778281, 26.74767524011239, 0],            # location-long
[8342, float, 55.036999999999999, 71.626000000000005, 66.53757922355274, 3.4426639342989835, 0],               # location-lat
[1, str, None, None, None, None, None],                                                                        # algorithm-marked-outlier
[219, float, 0.0, 3314.0, 6.77235908778888, 51.769138071783665, 0],                                            # argos:altitude
[29, float, -140.0, -109.0, -130.74352833498588, 3.1520512971719423, 0],                                       # argos:best-level
[26209, float, 401.6395579, 401.67991060000003, 401.65787084285711, 0.014282382007899122, 0],                  # argos:calcul-freq
[38, int, 0, 68, 42.067134467241246, 26.620460434692145, 0],                                                   # argos:iq
[8776, float, 46.021000000000001, 89.882999999999996, 66.583555731828241, 3.6088247789653005, 0],              # argos:lat1
[31336, float, -7.992, 89.974000000000004, 68.337729555335201, 12.003869177153453, 0],                         # argos:lat2
[7, str, None, None, None, None, None],                                                                        # argos:lc
[17225, float, -179.98699999999999, 179.929, -158.12360744413093, 31.182078273545585, 0],                      # argos:lon1
[56757, float, -179.995, 180.0, -21.033411440774923, 130.64698019622699, 0],                                   # argos:lon2
[15, int, 1, 15, 4.9746472427732442, 2.4039345107845147, 0],                                                   # argos:nb-mes
[5, int, 0, 4, 0.0033559606437342689, 0.069786351302635261, 0],                                                # argos:nb-mes-120
[5, int, 0, 4, 2.6839752879261689, 0.97545122273792406, 0],                                                    # argos:nopc
[728, float, 0.0, 999.0, 341.04696819464573, 169.37470497525266, 0],                                           # argos:pass-duration
[174, int, 0, 255, 142.03281214247579, 82.297515194018942, 0],                                                 # argos:sensor-1
[255, int, 0, 255, 155.06400732209596, 85.04541469161218, 0],                                                  # argos:sensor-2
[251, int, 0, 255, 65.396430478224389, 79.171224147694574, 0],                                                 # argos:sensor-3
[257, str, None, None, None, None, None],                                                                      # argos:sensor-4
[3, str, None, None, None, None, None],                                                                        # argos:valid-location-manual
[1, str, None, None, None, None, None],                                                                        # manually-marked-outlier
[1, str, None, None, None, None, None],                                                                        # manually-marked-valid
[1, str, None, None, None, None, None],                                                                        # sensor-type
[1, str, None, None, None, None, None],                                                                        # individual-taxon-canonical-name
[78, int, 2521, 41691, 21281.058592021967, 14214.950019017622, 0],                                             # tag-local-identifier
[110, str, None, None, None, None, None],                                                                      # individual-local-identifier
[1, str, None, None, None, None, None],                                                                        # study-name
[6017, float, -0.002839196, 0.001169741, -0.00020923727242201201, 0.00025372418999361626, 0],                  # ECMWF Interim Full Daily SFC-FC Evaporation
[65547, float, 241.40554729999999, 292.62299890000003, 273.64484250390967, 8.3539664894566119, 0],             # ECMWF Interim Full Daily SFC Temperature (2 m above Ground)
[45189, float, 0.0, 42333.0, 7521.4559724883902, 8667.6764867965921, 0],                                       # ECMWF Interim Full Daily SFC-FC Sunshine Duration
[65306, float, 4.4473302139999999, 20023.396130000001, 12498.842840947857, 7303.4622922395283, 0],             # NCEP NARR SFC Visibility at Surface
[51157, float, 0.010075927, 0.065644897999999993, 0.017614277201510181, 0.002483761423055314, 0],              # ECMWF Interim Full Daily SFC Charnock Parameter
[65551, float, -21.080586709999999, 20.16687464, -1.0247233927469757, 4.8647953166610574, 0],                  # ECMWF Interim Full Daily PL V Velocity
[65551, float, -20.810700090000001, 20.24135953, -1.45063101210373, 5.2434509217226912, 0],                    # ECMWF Interim Full Daily PL U Velocity
[60327, float, 9.9999999999999998e-13, 1.0, 0.743102277597807, 0.29391160015624507, 0],                        # ECMWF Interim Full Daily SFC Total Cloud Cover
[65513, float, 94959.482359999995, 104380.62820000001, 100349.69809264252, 1158.5317688474115, 0],             # ECMWF Interim Full Daily SFC Surface Air Pressure
[65551, float, 0.70862954, 39.72950247, 11.194576972507939, 6.4196639083671263, 0],                            # ECMWF Interim Full Daily SFC Total Atmospheric Water
[65543, float, 241.0419503, 290.97467289999997, 273.1702382199237, 7.5404664544593087, 0],                     # ECMWF Interim Full Daily SFC Snow Temperature
[59658, float, -34.62561917, 132.37799050000001, 4.6072516428248953, 7.9962651401687506, 0],                   # NASA Distance to Coast (Signed)
[16, int, 60, 230, 203.13950118221342, 20.229768280919515, 0],                                                 # GlobCover 2009 2009 Land-Cover Classification
[4350, float, 0.0, 0.0057368339999999997, 8.0336033771768732e-05, 0.00022090229077901508, 0],                  # ECMWF Interim Full Daily SFC-FC Runoff
[44104, float, 247.408705, 273.16082, 270.4825847814156, 3.3151211856370324, 0],                               # ECMWF Interim Full Daily SFC Ice Temperature at 0-7 cm
[65535, float, 248.04584869999999, 291.0741865, 273.84102653578526, 6.9635705954699905, 0],                    # ECMWF Interim Full Daily SFC Soil Temperature at 1-7 cm
[28228, float, 0.0, 1.0, 0.64421309954882156, 0.41041479238520817, 0],                                         # NCEP NARR SFC Snow Cover at Surface
[63902, float, -9.2599999999999993e-22, 0.36346331700000001, 0.09168344566862939, 0.069889765193965211, 0],    # ECMWF Interim Full Daily SFC Volumetric Soil Water Content at 1-7 cm
[-1, float, nan, nan, nan, nan, 56899],                                                                        # NCEP NARR 3D Cloud Water
[-1, float, nan, nan, nan, nan, 55580],                                                                        # ECMWF Interim Full Daily SFC Sea Ice Cover
[-1, float, nan, nan, nan, nan, 57270]                                                                         # SEDAC GRUMP v1 2000 Population Density Adjusted
]

TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
def parse_time(data):
    return datetime.datetime.strptime('{}000'.format(data), TIME_FORMAT)

def _parse_row(row):
    for i, val in enumerate(row):
        try:
            row[i] = int(val)
        except:
            try:
                row[i] = float(val)
            except:
                try:
                    row[i] = parse_time(val)
                except:
                    continue

    return row

def get_cols(file_name):
    with open(file_name, 'r') as f:
        first_line = f.readline()

    cols = [c.strip() for c in first_line.split(',')]
    cols = [c[1:-1] if c.startswith('"') else c for c in cols]
    return cols


def print_metadata(file_name):
    col_titles = get_cols(file_name)

    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter = ',')
        rows = []

        for index, row in enumerate(reader):
            if index == 0:
                continue
            row = _parse_row(row)
            rows.append(row)

        meta_data = []
        print "Finished reading data"

        for col_index in xrange(len(rows[0])):
            all_col = [row[col_index] for row in rows]

            # Carefully inspect the data to correctly identify the data type
            data_type = [type(row[col_index]) for row in rows]

            if int in data_type:
                if float in data_type:
                    data_type = float
                elif len(set(data_type)) == 1:
                    data_type = int
                else:
                    data_type = str
            else:
                data_type = data_type[0]

            if data_type is float and np.isnan(all_col).any():
                occurences = -1
            else:
                occurences = len(set(all_col))

            minn, maxx, avg, std, nan_count = None, None, None, None, None

            if data_type in [int, float]:
                try:
                    minn = np.min(all_col)
                    maxx = np.max(all_col)
                    avg = np.average(all_col)
                    std = np.std(all_col)
                    nan_count = np.count_nonzero(np.isnan(all_col))
                except:
                    minn, maxx, avg, std, nan_count = None, None, None, None, None

            data_type_str = str(data_type)[len('<type \''):-2]
            new_meta = [occurences, data_type_str, minn, maxx, avg, std, nan_count]
            print '{0:112} # {1}'.format('%s,' % new_meta, col_titles[col_index])
