import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.coordinates import SkyCoord
from astropy import units as u
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull
import copy

# TODO
# Настройка угла соединения филаментов в рассчете метрик
# Переменные указывающие было ли выполненно некоторое действие
# Перевод из декартовых в сферические
# Куда сохранять случайные кластера?


def dist(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2))**0.5


def intersec_line_sphere(x1, y1, z1, x2, y2, z2, x3, y3, z3, r):
    a = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
    b = 2 * ((x2-x1)*(x1-x3) + (y2-y1)*(y1-y3) + (z2-z1)*(z1-z3))
    c = x3**2 + y3**2 + z3**2 + x1**2 + y1**2 + z1**2 \
        - 2 * (x3*x1 + y3*y1 + z3*z1) - r**2
    d = b**2 - 4*a*c
    if d < 0:
        return False
    if d == 0:
        u_ = -b/(2*a)
        # print(f'a={a}, b={b}, u_={u_}')
        if 0 <= u_ <= 1:
            return True
        else:
            return False
    if d > 0:
        u1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        u2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
        if (0 <= u1 <= 1) or (0 <= u2 <= 1):
            return True
        if (u1 <= 0) and (u2 >= 1) or (u2 <= 0) and (u1 >= 1):
            return True
        return False
    print(d, 'intersec_line_sphere_ERROR')


class Disperse3D:
    RANDOM_CLUSTERS_NUM = 5
    random_clusters = None

    def __init__(
        self, galaxies, clusters, disperse_path,
        cosmo_H0, cosmo_Om, cosmo_Ol, cosmo_Ok,
        sph2cart_f='dist', cart2sph_f='dist'
    ):
        #GALAXIES and CLUSTERS must have fields 'RA', 'DEC', 'Z'

        self.disperse_path = disperse_path
        if self.disperse_path[-1] != '/':
            self.disperse_path += '/'
        self.galaxies = galaxies.copy()
        self.clusters = clusters.copy()
        self.ra_int = (self.galaxies['RA'].min() - 5, self.galaxies['RA'].max() + 5)
        self.dec_int = (self.galaxies['DEC'].min() - 5, self.galaxies['DEC'].max() + 5)
        self.z_int = (self.galaxies['Z'].min() - 0.002, self.galaxies['Z'].max() + 0.002)
        self.cosmo_H0 = cosmo_H0 / 100
        self.cosmo_Om = cosmo_Om
        self.cosmo_Ol = cosmo_Ol
        self.cosmo_Ok = cosmo_Ok
        self.cosmo = FlatLambdaCDM(H0=cosmo_H0, Om0=cosmo_Om)
        self.COORDS_IN = f'{id(self)}_coords_ascii.txt'
        self.DISPERSE_IN = f'{id(self)}_galaxies_ascii.txt'
        if sph2cart_f == 'min':
            self.sph2cart = self.sph2cart_DTFE_MIN
        elif sph2cart_f == 'dist':
            self.sph2cart = self.sph2cart_DIST
        elif sph2cart_f == 'astropy':
            self.sph2cart = self.sph2cart_ASTROPY
        else:
            print('WRONG shp2cart_f value')
            self.sph2cart = self.sph2cart_DIST
            
        #TODO
        if cart2sph_f == 'dist':
            self.cart2sph = self.cart2sph_DIST
        elif cart2sph_f == 'astropy':
            self.cart2sph = self.cart2sph_ASTROPY
        else:
            print('WRONG cart2sph_f value')
            self.cart2sph = self.cart2sph_DIST

        self.cart_coords = False
        self.apply = False

        self.fils = None

        self.disperse_sigma = None
        self.disperse_smooth = None
        self.disperse_board = None
        self.disprese_asmb_angle = None

        self.metrics = None

    def sph2cart_DTFE_MIN(self, ra, dec, z):
        uniq_a = []
        uniq_d = {}
        for i in range(len(ra)):
            t = (ra[i], dec[i], z[i])
            if t not in uniq_d:
                uniq_d[t] = None
                uniq_a.append(t)
        with open(f'{self.COORDS_IN}', 'w') as f:
            f.write('# ra dec z\n')
            for i in range(len(uniq_a)):
                f.write(f'{uniq_a[i][0]}\t{uniq_a[i][1]}\t{uniq_a[i][2]}\n')
        os.system((
            f'{self.disperse_path}delaunay_3D {self.COORDS_IN} '
            f'-btype void  -minimal '
            f'-cosmo {self.cosmo_Om} {self.cosmo_Ol} {self.cosmo_Ok} {self.cosmo_H0} {-1.0}'
        ))
        os.system(
            f'{self.disperse_path}netconv {self.COORDS_IN}.NDnet -to NDnet_ascii'
        )

        CX, CY, CZ = [], [], []
        with open(f'{self.COORDS_IN}.NDnet.a.NDnet', 'r') as f:
            for i in range(4):
                f.readline()
            n = int(f.readline())
            if n != len(uniq_a):
                print('ERROR!')
                return
            for i in range(n):
                cx, cy, cz = tuple(map(float, f.readline().split()))
                uniq_d[uniq_a[i]] = (cx, cy, cz)

        for i in range(len(ra)):
            t = (ra[i], dec[i], z[i])
            t = uniq_d[t]
            CX.append(t[0])
            CY.append(t[1])
            CZ.append(t[2])

        os.system(f'rm {self.COORDS_IN}*')

        return CX, CY, CZ

    def sph2cart_DIST(self, ra, dec, z):
        with open(self.COORDS_IN, 'w') as f:
            for i in range(len(z)):
                f.write(f'{z[i]}\n')
        os.system((
            f'{self.disperse_path}my_dist '
            f'{self.cosmo_Om} {self.cosmo_Ol} 0.0 {self.cosmo_H0} '
            f'{self.COORDS_IN} out_{self.COORDS_IN} s'
        ))
        dist = []
        with open(f'out_{self.COORDS_IN}', 'r') as f:
            for line in f:
                dist.append(float(line))
        os.system(
            f'rm {self.COORDS_IN} out_{self.COORDS_IN}'
        )

        CX, CY, CZ = [], [], []
        for i in range(len(ra)):
            x = dist[i] * np.cos(ra[i] * np.pi / 180) * np.cos(dec[i] * np.pi / 180)
            y = dist[i] * np.sin(ra[i] * np.pi / 180) * np.cos(dec[i] * np.pi / 180)
            z = dist[i] * np.sin(dec[i] * np.pi / 180)
            CX.append(x)
            CY.append(y)
            CZ.append(z)

        return CX, CY, CZ

    #TODO
    def cart2sph_DIST(self, CX, CY, CZ):
        ra, dec, dist = [], [], []
        for i in range(len(CX)):
            ra.append(np.arctan(CY[i] / CX[i]))
            dec.append(np.arctan((CX[i]**2 + CY[i]**2)**0.5 / CZ[i]))
            dist.append((CX[i]**2 + CY[i]**2 + CZ[i]**2)**0.5)

        with open(self.COORDS_IN, 'w') as f:
            for i in range(len(dist)):
                f.write(f'{dist[i]}\n')
        os.system((
            f'{self.disperse_path}my_dist '
            f'{self.cosmo_Om} {self.cosmo_Ol} 0.0 {self.cosmo_H0} '
            f'{self.COORDS_IN} out_{self.COORDS_IN} c'
        ))
        z = []
        with open(f'out_{self.COORDS_IN}', 'r') as f:
            for line in f:
                z.append(float(line))
        os.system(
            f'rm {self.COORDS_IN} out_{self.COORDS_IN}'
        )

        return ra, dec, z

    def sph2cart_ASTROPY(self, ra, dec, z):
        CX, CY, CZ = [], [], []
        for i in range(len(ra)):
            c = SkyCoord(
                ra=ra[i],
                dec=dec[i],
                distance=self.cosmo.comoving_distance(z[i]),
                frame='fk5'
            )
            c.representation_type = 'cartesian'
            CX.append(c.x.value)
            CY.append(c.y.value)
            CZ.append(c.z.value)

        return CX, CY, CZ

    #TODO
    def cart2shp_ASTROPY(self, CX, CY, CZ):
        ra, dec, z = [], [], []
        for i in range(len(CX)):
            c = SkyCoord(
                x=CX[i],
                y=CY[i],
                z=CZ[i],
                frame='fk5',
                unit='Mpc',
                representation_type='cartesian'
            )
            c.representation_type = 'spherical'
            ra.append(c.ra.value)
            dec.append(c.dec.value)
            z.append(z_at_value(self.cosmo.comoving_distance, c.distance))

        return ra, dec, z

    def count_cart_coords(self):
        CX, CY, CZ = self.sph2cart(
            self.galaxies['RA'], self.galaxies['DEC'], self.galaxies['Z']
        )
        self.galaxies = self.galaxies.assign(CX=CX)
        self.galaxies = self.galaxies.assign(CY=CY)
        self.galaxies = self.galaxies.assign(CZ=CZ)

        CX, CY, CZ = self.sph2cart(
            self.clusters['RA'], self.clusters['DEC'], self.clusters['Z']
        )
        self.clusters = self.clusters.assign(CX=CX)
        self.clusters = self.clusters.assign(CY=CY)
        self.clusters = self.clusters.assign(CZ=CZ)

        self.cart_coords = True

    #TODO
    def count_sph_coords(self):
        ra, dec, z = self.cart2sph(
            self.galaxies['CX'], self.galaxies['CY'], self.galaxies['CZ']
        )
        self.galaxies = self.galaxies.assign(RA=ra)
        self.galaxies = self.galaxies.assign(DEC=dec)
        self.galaxies = self.galaxies.assign(Z=z)

        ra, dec, z = self.cart2sph(
            self.clusters['RA'], self.clusters['DEC'], self.clusters['Z']
        )
        self.clusters = self.clusters.assign(RA=ra)
        self.clusters = self.clusters.assign(DEC=dec)
        self.clusters = self.clusters.assign(Z=z)

    def apply_disperse(
        self, disperse_sigma, disperse_smooth, disperse_board='smooth',
        disprese_asmb_angle=30, in_cart_coords=True
    ):
        self.fils = None
        self.metrics = None

        self.disperse_sigma = disperse_sigma if disperse_sigma != int(disperse_sigma) else int(disperse_sigma)
        self.disperse_smooth = disperse_smooth
        self.disperse_board = disperse_board
        self.disprese_asmb_angle = disprese_asmb_angle

        if in_cart_coords:
            with open(self.DISPERSE_IN, 'w') as f:
                f.write('# px py pz\n')
                for i in range(self.galaxies.shape[0]):
                    t = self.galaxies.iloc[i]
                    f.write(f'{t.CX}\t{t.CY}\t{t.CZ}\n')
        else:
            with open(self.DISPERSE_IN, 'w') as f:
                f.write('# ra dec z\n')
                for i in range(self.galaxies.shape[0]):
                    t = self.galaxies.iloc[i]
                    f.write(f'{t.RA}\t{t.DEC}\t{t.Z}\n')

        os.system((
            f'{self.disperse_path}delaunay_3D {self.DISPERSE_IN} '
            f'-btype {self.disperse_board} -smooth {self.disperse_smooth} '
            f'-cosmo {self.cosmo_Om} {self.cosmo_Ol} {self.cosmo_Ok} {self.cosmo_H0} {-1.0}'
        ))

        os.system((
            f'{self.disperse_path}mse {self.DISPERSE_IN}.NDnet '
            f'-upSkl -forceLoops -nsig {self.disperse_sigma}'
        ))

        os.system((
            f'{self.disperse_path}skelconv {self.DISPERSE_IN}.NDnet_s{self.disperse_sigma}.up.NDskl '
            f'-breakdown -to NDskl_ascii {f"-assemble 0 {self.disprese_asmb_angle}"} -toRaDecZ '
            f'-cosmo {self.cosmo_Om} {self.cosmo_Ol} {self.cosmo_Ok} {self.cosmo_H0} {-1.0}'
        ))

        self.read_skl_ascii_RaDecZ(
            f'{self.DISPERSE_IN}.NDnet_s{self.disperse_sigma}.up.NDskl.BRK.ASMB.RaDecZ.a.NDskl'
        )

        os.system(f'rm {self.DISPERSE_IN}* test_smooth.dat')

        self.apply=True

    def read_skl_ascii_RaDecZ(self, file_name):
        self.cps = []
        self.fils = []
        with open(file_name) as f:
            s = ''
            while s != '[CRITICAL POINTS]':
                s = f.readline().strip()
            cp_num = int(f.readline().strip())
            ras = []
            decs = []
            zs = []
            types = []
            values = []
            for i in range(cp_num):
                type_, ra, dec, z, value, _, _ = tuple(map(float, f.readline().split()))
                type_ = int(type_)
                ras.append(ra)
                decs.append(dec)
                zs.append(z)
                types.append(type_)
                values.append(value)
                for j in range(int(f.readline())):
                    f.readline()
            cx, cy, cz = self.sph2cart(ras, decs, zs)
            for i in range(cp_num):
                self.cps.append({
                    'RA': ras[i], 'DEC': decs[i], 'Z': zs[i],
                    'CX': cx[i], 'CY': cy[i], 'CZ': cz[i],
                    'type': types[i], 'value': values[i]
                })

            while s != '[FILAMENTS]':
                s = f.readline().strip()
            fil_num = int(f.readline())
            ras = []
            decs = []
            zs = []
            for i in range(fil_num):
                fil = {}
                cp1, cp2, sp_num = tuple(map(int, f.readline().split()))
                fil['CP1_id'] = cp1
                fil['CP2_id'] = cp2
                fil['sp_num'] = sp_num
                fil['sample_points'] = []
                for j in range(sp_num):
                    ra, dec, z = tuple(map(float, f.readline().split()))
                    ras.append(ra)
                    decs.append(dec)
                    zs.append(z)
                self.fils.append(fil)
            cx, cy, cz = self.sph2cart(ras, decs, zs)
            k = 0
            for i in range(fil_num):
                for j in range(self.fils[i]['sp_num']):
                    self.fils[i]['sample_points'].append({
                        'RA': ras[k + j], 'DEC': decs[k + j], 'Z': zs[k + j],
                        'CX': cx[k + j], 'CY': cy[k + j], 'CZ': cz[k + j]
                    })
                k += self.fils[i]['sp_num']

        self.maxs = []
        for cp in self.cps:
            if cp['type'] == 3:
                self.maxs.append(cp.copy())
        self.maxs = sorted(self.maxs, key=lambda x: -x['value'])

    def count_conn(self, cl_conn_rads, clusters=None):
        if clusters is None:
            clusters = self.clusters

        MIN_SEG_LEN = 1  # Mpc

        points = []
        next_point = []
        fil_num = []
        count = 0
        for i, fil in enumerate(self.fils):
            sp = fil['sample_points']
            for j in range(len(sp) - 1):
                points.append([sp[j]['CX'], sp[j]['CY'], sp[j]['CZ']])
                fil_num.append(i)
                count += 1
                next_point.append(count)
                d = dist(
                    sp[j]['CX'], sp[j]['CY'], sp[j]['CZ'],
                    sp[j + 1]['CX'], sp[j + 1]['CY'], sp[j + 1]['CZ']
                )
                if d > MIN_SEG_LEN:
                    n = int(d // MIN_SEG_LEN + 1)
                    d_x = sp[j + 1]['CX'] - sp[j]['CX']
                    d_y = sp[j + 1]['CY'] - sp[j]['CY']
                    d_z = sp[j + 1]['CZ'] - sp[j]['CZ']
                    for k in range(1, n):
                        points.append([sp[j]['CX'] + k * d_x / n, sp[j]['CY'] + k * d_y / n, sp[j]['CZ'] + k * d_z / n])
                        fil_num.append(i)
                        count += 1
                        next_point.append(count)
            points.append([sp[-1]['CX'], sp[-1]['CY'], sp[-1]['CZ']])
            fil_num.append(i)
            count += 1
            next_point.append(None)

        kd_tree = KDTree(points, leaf_size=2)

        cl_conn = [0] * clusters.shape[0]
        fils_conn = [0] * len(self.fils)

        CX = clusters['CX']
        CY = clusters['CY']
        CZ = clusters['CZ']

        for i in range(self.clusters.shape[0]):
            x3 = CX[i]
            y3 = CY[i]
            z3 = CZ[i]
            r_fil = cl_conn_rads[i]

            proc_fils = set()

            close_points_idx = kd_tree.query_radius([[x3, y3, z3]], r=r_fil + MIN_SEG_LEN + 1)
            for p_idx in close_points_idx[0]:
                if fil_num[p_idx] in proc_fils:
                    continue
                if next_point[p_idx] is None:
                    continue
                x1, y1, z1 = tuple(points[p_idx])
                x2, y2, z2 = tuple(points[next_point[p_idx]])
                if intersec_line_sphere(
                    x1, y1, z1,
                    x2, y2, z2,
                    x3, y3, z3,
                    r_fil
                ):
                    cl_conn[i] += 1
                    fils_conn[fil_num[p_idx]] += 1
                    proc_fils.add(fil_num[p_idx])

        # for i in range(self.clusters.shape[0]):
        #     x3 = CX[i]
        #     y3 = CY[i]
        #     z3 = CZ[i]
        #     r_fil = cl_conn_rads[i]
        #     r_max = cl_maxmap_rads[i]
        #     for j, fil in enumerate(self.fils):
        #         points = fil['sample_points']
        #         for l in range(len(points) - 1):
        #             x1, y1, z1 = points[l]['CX'], points[l]['CY'], points[l]['CZ']
        #             x2, y2, z2 = points[l + 1]['CX'], points[l + 1]['CY'], points[l + 1]['CZ']
        #             if intersec_line_sphere(
        #                     x1, y1, z1,
        #                     x2, y2, z2,
        #                     x3, y3, z3,
        #                     r_fil
        #             ):
        #                 cl_conn[i] += 1
        #                 fils_conn[j] += 1
        #                 count_fils += 1
        #                 break
        #
        #     # for j, m in enumerate(self.maxs):
        #     #     if dist(m['CX'], m['CY'], m['CZ'],
        #     #             x3, y3, z3) <= r_max:
        #     #         cl_maxmap[i] += 1
        #     #         maxs_maxmap[j] += 1
        #     #         count_maxs += 1
        #

        return cl_conn, fils_conn

    def count_metrics(self, mode, rads, clusters=None):
        if clusters is None:
            clusters = self.clusters
        if self.metrics is not None:
            print('Metrics was already computed')
            return
        if not self.apply:
            print('DisPerSe wasn\'t computed')
            return
        if Disperse3D.random_clusters is None or \
                Disperse3D.random_clusters[0].shape[0] != clusters.shape[0]:
            CX_int = (self.galaxies['CX'].min(), self.galaxies['CX'].max())
            CY_int = (self.galaxies['CY'].min(), self.galaxies['CY'].max())
            CZ_int = (self.galaxies['CZ'].min(), self.galaxies['CZ'].max())

            points = np.array(self.galaxies[['CX', 'CY', 'CZ']])
            hull = ConvexHull(points)
            A, b = hull.equations[:, :-1], hull.equations[:, -1:]
            EPS = -5

            def contained(x):
                return np.all(np.asarray(x) @ A.T + b.T < EPS, axis=1)

            np.random.seed(0)

            for i in range(Disperse3D.RANDOM_CLUSTERS_NUM):
                CX, CY, CZ = [], [], []
                for j in tqdm(range(clusters.shape[0])):
                    fl = False
                    while not fl:
                        cx = np.random.uniform(CX_int[0], CX_int[1], 1)[0]
                        cy = np.random.uniform(CY_int[0], CY_int[1], 1)[0]
                        cz = np.random.uniform(CZ_int[0], CZ_int[1], 1)[0]
                        fl = contained([[cx, cy, cz]])
                    CX.append(cx)
                    CY.append(cy)
                    CZ.append(cz)
                df = pd.DataFrame()
                df = df.assign(CX=CX)
                df = df.assign(CY=CY)
                df = df.assign(CZ=CZ)
                ra, dec, z = self.cart2shp_ASTROPY(CX, CY, CZ)
                df = df.assign(RA=ra)
                df = df.assign(DEC=dec)
                df = df.assign(Z=z)
                Disperse3D.random_clusters.append(df)

        self.metrics = {}
        self.metrics['sigma'] = self.disperse_sigma
        self.metrics['smooth'] = self.disperse_smooth
        self.metrics['angle'] = self.disprese_asmb_angle
        self.metrics['mode'] = mode
        self.metrics['rads'] = rads

        cl_num = clusters.shape[0]
        fils_num = len(self.fils)

        true_cl_inter = []
        true_fils_inter = []
        for rad in tqdm(rads):
            if mode == 'coefs':
                cl_conn, fils_conn = DPS.count_conn(
                    self.clusters['R'] * rad,
                    clusters
                )
            else:
                cl_conn, fils_conn = DPS.count_conn(
                    [rad] * self.clusters.shape[0],
                    clusters
                )
            true_cl_inter.append(sum(list(map(lambda x: int(x > 0), cl_conn))))
            true_fils_inter.append(sum(list(map(lambda x: int(x > 0), fils_conn))))
        true_cl_inter = np.array(true_cl_inter)
        true_fils_inter = np.array(true_fils_inter)

        false_cl_inter = []
        false_fils_inter = []
        for i in tqdm(range(Disperse3D.RANDOM_CLUSTERS_NUM)):
            false_cl_inter.append([])
            false_fils_inter.append([])
            for rad in rads:
                if mode == 'coefs':
                    cl_conn, fils_conn = DPS.count_conn(
                        Disperse3D.random_clusters[i]['R'] * rad
                    )
                else:
                    cl_conn, fils_conn = DPS.count_conn(
                        [rad] * Disperse3D.random_clusters[i]['R']
                    )
                false_cl_inter[i].append(sum(list(map(lambda x: int(x > 0), cl_conn))))
                false_fils_inter[i].append(sum(list(map(lambda x: int(x > 0), fils_conn))))

        false_cl_inter = np.array(false_cl_inter).mean(0)
        false_fils_inter = np.array(false_fils_inter).mean(0)

        diff_cl_inter = true_cl_inter - false_cl_inter
        diff_fils_inter = true_fils_inter - false_fils_inter

        true_recall = true_cl_inter / cl_num
        false_recall = false_cl_inter / cl_num
        diff_recall = diff_cl_inter / cl_num

        true_precision = true_fils_inter / fils_num
        false_precision = false_fils_inter / fils_num
        diff_precision = diff_fils_inter / fils_num

        true_f1 = 2 * true_recall * true_precision / (true_recall + true_precision)
        false_f1 = 2 * false_recall * false_precision / (false_recall + false_precision)
        diff_f1 = 2 * diff_recall * diff_precision / (diff_recall + diff_precision)

        self.metrics['cl_num'] = cl_num
        self.metrics['fils_num'] = fils_num

        self.metrics['true_cl_inter'] = [int(e) for e in true_cl_inter]
        self.metrics['false_cl_inter'] = [float(e) for e in false_cl_inter]
        self.metrics['diff_cl_inter'] = [float(e) for e in diff_cl_inter]

        self.metrics['true_fils_inter'] = [int(e) for e in true_fils_inter]
        self.metrics['false_fils_inter'] = [int(e) for e in false_fils_inter]
        self.metrics['diff_fils_inter'] = [int(e) for e in diff_fils_inter]

        self.metrics['true_recall'] = [int(e) for e in true_recall]
        self.metrics['false_recall'] = [float(e) for e in false_recall]
        self.metrics['diff_recall'] = [float(e) for e in diff_recall]

        self.metrics['true_precision'] = [int(e) for e in true_precision]
        self.metrics['false_precision'] = [float(e) for e in false_precision]
        self.metrics['diff_precision'] = [float(e) for e in diff_precision]

        self.metrics['true_f1'] = [int(e) for e in true_f1]
        self.metrics['false_f1'] = [float(e) for e in false_f1]
        self.metrics['diff_f1'] = [float(e) for e in diff_f1]

    def count_metrics_several_params(self, sigmas, smooths, mode, rads, clusters=None):
        if clusters is None:
            clusters = self.clusters
        metrics = {}
        metrics['sigmas'] = sigmas
        metrics['smooths'] = smooths
        metrics['mode'] = mode
        metrics['rads'] = rads
        for sigma in sigmas:
            metrics[sigma] = {}
            for smooth in smooths:
                self.apply_disperse(sigma, smooth)
                self.count_metrics(mode, rads, clusters)
                metrics[sigma][smooth] = copy.deepcopy(self.metrics)

        return metrics

    def plot_2d(
        self, plot_galaxies=True, plot_clusters=True,
        plot_cps=True, plot_only_max=True, plot_fils=True,
        cl_fils=None, cl_maxs=None, title=None, clusters=None
    ):
        if clusters is None:
            clusters = self.clusters

        font = {'size': 16}
        plt.rc('font', **font)
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(111)
        print(ax)

        if plot_galaxies:
            ax.scatter(self.galaxies['RA'], self.galaxies['DEC'], c='grey', s=8)

        if plot_clusters:
            ax.scatter(clusters['RA'], clusters['DEC'], c='purple', s=150)
            if cl_maxs is not None:
                t = clusters[cl_maxs]
                ax.scatter(
                    t['RA'], t['DEC'],
                    marker='s', facecolors='none', edgecolors='orange', linewidths=5, s=500
                )
            if cl_fils is not None:
                t = clusters[cl_fils]
                ax.scatter(
                    t['RA'], t['DEC'],
                    facecolors='none', edgecolors='cyan', linewidths=4, s=300
                )

        if plot_cps:
            d = {4: 'xkcd:brown', 3: 'red', 2: 'green', 1: 'orange', 0: 'blue'}
            x, y, c = [], [], []
            for cp in self.cps:
                if cp['type'] != 3 and plot_only_max:
                    continue
                x.append(cp['RA'])
                y.append(cp['DEC'])
                c.append(d[cp['type']])
            ax.scatter(x, y, c=c, s=100)

        if plot_fils:
            for fil in self.fils:
                points = fil['sample_points']
                x = []
                y = []
                for i in range(len(points)):
                    x.append(points[i]['RA'])
                    y.append(points[i]['DEC'])
                ax.plot(x, y, 'b', linewidth=2, color='b')

        ax.invert_xaxis()
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        if title is None:
            title = (
                f'DisPerSe_2D_smooth:{self.disperse_smooth}_s:{self.disperse_sigma}_'
                f'board:{self.disperse_board}_asmb:{self.disprese_asmb_angle}'
            )
        ax.set_title(title)

        return fig, title

    def plot_3d(
        self, plot_galaxies=False, plot_clusters=True,
        plot_cps=False, plot_only_max=True, plot_fils=True,
        cl_fils=None, cl_maxs=None, title=None, clusters=None
    ):
        if clusters is None:
            clusters = self.clusters

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(self.ra_int)
        ax.set_zlim(self.dec_int)
        ax.set_ylim(self.z_int)

        if plot_galaxies:
            ax.scatter(
                self.galaxies['RA'], self.galaxies['Z'], self.galaxies['DEC'],
                c='grey', s=2, alpha=0.3
            )

        if plot_clusters:
            ax.scatter(
                clusters['RA'], clusters['Z'], clusters['DEC'],
                color='purple', s=40, alpha=1
            )
            if cl_maxs is not None:
                t = clusters[cl_maxs]
                ax.scatter(
                    t['RA'], t['Z'], t['DEC'],
                    marker='s', facecolors='none', edgecolors='orange', linewidths=5, s=50
                )
            if cl_fils is not None:
                t = clusters[cl_fils]
                ax.scatter(
                    t['RA'], t['Z'], t['DEC'],
                    facecolors='none', edgecolors='cyan', linewidths=2, s=30
                )

        if plot_cps:
            d = {4: 'xkcd:brown', 3: 'red', 2: 'green', 1: 'orange', 0: 'blue'}
            x = []
            y = []
            z = []
            c = []
            for cp in self.cps:
                if cp['type'] != 3 and plot_only_max:
                    continue
                x.append(cp['RA'])
                y.append(cp['DEC'])
                z.append(cp['Z'])
                c.append(d[cp['type']])
            ax.scatter(x, z, y, c=c, s=10)

        if plot_fils:
            for fil in tqdm(self.fils):
                points = fil['sample_points']
                x = []
                y = []
                z = []
                for i in range(len(points)):
                    x.append(points[i]['RA'])
                    y.append(points[i]['DEC'])
                    z.append(points[i]['Z'])
                ax.plot(x, z, y, 'b', linewidth=1, color='b')

        ax.invert_xaxis()
        ax.set_xlabel('RA')
        ax.set_ylabel('Z')
        ax.set_zlabel('DEC')
        if title is None:
            title = (
                f'DisPerSe_3D_smooth:{self.disperse_smooth}_s:{self.disperse_sigma}_'
                f'board:{self.disperse_board}_asmb:{self.disprese_asmb_angle}'
            )
        ax.set_title(title)

        return fig, title
