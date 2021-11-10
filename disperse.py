import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


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
        u = -b/(2*a)
        if 0 <= u <= 1:
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


class Disperse:
    def __init__(
        self, galaxies, clusters, disperse_path,
        cosmo_H0, cosmo_Om, cosmo_Ol, cosmo_Ok
    ):
        self.disperse_path = disperse_path
        if self.disperse_path[-1] != '/':
            self.disperse_path += '/'
        self.galaxies = galaxies
        self.clusters = clusters
        self.cosmo_H0 = cosmo_H0 / 100
        self.cosmo_Om = cosmo_Om
        self.cosmo_Ol = cosmo_Ol
        self.cosmo_Ok = cosmo_Ok
        self.COORDS_IN = f'{id(self)}_coords_ascii.txt'
        self.DISPERSE_IN = f'{id(self)}_galaxies_ascii.txt'

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

    def count_cart_coords(self):
        CX, CY, CZ = self.sph2cart_DTFE_MIN(
            self.galaxies['RA'], self.galaxies['DEC'], self.galaxies['Z']
        )
        self.galaxies = self.galaxies.assign(CX=CX)
        self.galaxies = self.galaxies.assign(CY=CY)
        self.galaxies = self.galaxies.assign(CZ=CZ)

        CX, CY, CZ = self.sph2cart_DTFE_MIN(
            self.clusters['RA'], self.clusters['DEC'], self.clusters['Z']
        )
        self.clusters = self.clusters.assign(CX=CX)
        self.clusters = self.clusters.assign(CY=CY)
        self.clusters = self.clusters.assign(CZ=CZ)

    def apply_disperse(
        self, disperse_sigma, disperse_smooth, disperse_board
    ):
        self.disperse_sigma = disperse_sigma
        self.disperse_smooth = disperse_smooth
        self.disperse_board = disperse_board

        with open(self.DISPERSE_IN, 'w') as f:
            f.write('# px py pz\n')
            for i in range(self.galaxies.shape[0]):
                t = self.galaxies.iloc[i]
                f.write(f'{t.CX}\t{t.CY}\t{t.CZ}\n')

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
            f'-breakdown -to NDskl_ascii -toRaDecZ '
            f'-cosmo {self.cosmo_Om} {self.cosmo_Ol} {self.cosmo_Ok} {self.cosmo_H0} {-1.0}'
        ))

        self.read_skl_ascii_RaDecZ(
            f'{self.DISPERSE_IN}.NDnet_s{self.disperse_sigma}.up.NDskl.BRK.RaDecZ.a.NDskl'
        )

        os.system(f'rm {self.DISPERSE_IN}* test_smooth.dat')

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
            cx, cy, cz = self.sph2cart_DTFE_MIN(ras, decs, zs)
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
            cx, cy, cz = self.sph2cart_DTFE_MIN(ras, decs, zs)
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

    def count_conn_maxmap(self, cl_conn_rads, cl_maxmap_rads):
        cl_conn = [0] * self.clusters.shape[0]
        cl_maxmap = [0] * self.clusters.shape[0]

        fils_conn = [0] * len(self.fils)
        maxs_maxmap = [0] * len(self.maxs)

        count_fils = 0
        count_maxs = 0

        for i in range(self.clusters.shape[0]):
            x3 = self.clusters.iloc[i]['CX']
            y3 = self.clusters.iloc[i]['CY']
            z3 = self.clusters.iloc[i]['CZ']
            r_fil = cl_conn_rads[i]
            r_max = cl_maxmap_rads[i]
            for j, fil in enumerate(self.fils):
                points = fil['sample_points']
                for l in range(len(points) - 1):
                    x1, y1, z1 = points[l]['CX'], points[l]['CY'], points[l]['CZ']
                    x2, y2, z2 = points[l + 1]['CX'], points[l + 1]['CY'], points[l + 1]['CZ']
                    if intersec_line_sphere(
                            x1, y1, z1,
                            x2, y2, z2,
                            x3, y3, z3,
                            r_fil
                    ):
                        cl_conn[i] += 1
                        fils_conn[j] += 1
                        count_fils += 1
                        break

            for j, m in enumerate(self.maxs):
                if dist(m['CX'], m['CY'], m['CZ'],
                        x3, y3, z3) <= r_max:
                    cl_maxmap[i] += 1
                    maxs_maxmap[j] += 1
                    count_maxs += 1

        return cl_conn, cl_maxmap, \
               fils_conn, maxs_maxmap, \
               count_fils, count_maxs

    def plot_2d(
        self, plot_galaxies=True, plot_clusters=True,
        plot_cps=True, plot_only_max=True, plot_fils=True,
        cl_fils=None, cl_maxs=None
    ):
        font = {'size': 16}
        plt.rc('font', **font)
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(111)
        print(ax)

        if plot_galaxies:
            ax.scatter(self.galaxies['RA'], self.galaxies['DEC'], c='grey', s=8)

        if plot_clusters:
            ax.scatter(self.clusters['RA'], self.clusters['DEC'], c='purple', s=150)
            if cl_maxs is not None:
                t = self.clusters[cl_maxs]
                ax.scatter(
                    t['RA'], t['DEC'],
                    marker='s', facecolors='none', edgecolors='orange', linewidths=5, s=500
                )
            if cl_fils is not None:
                t = self.clusters[cl_fils]
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
                ax.plot(x, y, 'b', linewidth=1)

        ax.invert_xaxis()
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        ax.set_title(f'DisPerSe_smooth:{self.disperse_smooth}_s:{self.disperse_sigma}_board:{self.disperse_board}')

        return fig
