import json
import argparse
import scipy

import numpy as np
from matplotlib import pyplot as plt

def rotate(p, theta):
    
    x,y = p
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y,x)
    x_new = r*np.cos(th + theta)
    y_new = r*np.sin(th + theta)
    return [x_new, y_new]

def r_all(v, theta):
    return np.array([rotate(p, theta) for p in v])

def translate_all(v, delta):
    return np.array([p + delta for p in v])



def best_permute(v, ref):
    ret = []
    for p in v:
        ## Find closest point in ref to p
        i_best = 0
        x_p, y_p = p
        r_best = 9e99
        for i, orig in enumerate(ref):
            x_0, y_0 = orig
            r2 = (x_p-x_0)**2 + (y_p-y_0)**2
            if r2 < r_best:
                r_best = r2
                i_best = i
        ret.append(i_best)
    return ret

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(
        description="Analyze the positions and match antennas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--original",
        required=True,
        help="Original (uncalibrated) antenna positions.",
    )
    parser.add_argument(
        "--drone",
        required=False,
        default="antenna_positions.json",
        help="Input file for drone antenna positions.",
    )
    parser.add_argument(
        "--outfile",
        required=False,
        default="matched_antenna_positions.json",
        help="Output file for matched antenna positions.",
    )

    ARGS = parser.parse_args()

    # Opening JSON file
    with open(ARGS.drone) as f:
        drone = json.load(f)["antenna_positions"]
        
        drone = np.array(drone)
        mu = np.mean(drone, axis=0)
        drone = drone - mu
        
    with open(ARGS.original) as f:
        original = json.load(f)
        original = np.array(original)[:,0:2]
        mu = np.mean(original, axis=0)
        original = original - mu
        

    print(drone)
    # print(original)
    

    ## Find the rotation that matches best
    
    def f_min(x_deg):
        ret = 0.0
        _drone_r = r_all(drone, np.radians(x_deg))
        
        for p in _drone_r:
            ## Find closest point in original to p
            i_best = 0
            x_p, y_p = p
            r_best = 9e99
            for i, orig in enumerate(original):
                x_0, y_0 = orig
                r2 = (x_p-x_0)**2 + (y_p-y_0)**2
                if r2 < r_best:
                    r_best = r2
                    i_best = i
                    
            ret += r_best
            
        return ret

    # for th in np.linspace(-30,30,100):
    #     print(f"{th}, {f_min(th)}")

    ret = scipy.optimize.minimize(f_min, -40, method="BFGS", bounds=(-33,33))
    print(ret)

    angle = ret.x[0]

    drone_r = r_all(drone, np.radians(angle))
    print(f"Angle between drone and original = {angle} degrees")
    plt.plot(drone_r[:,0], drone_r[:,1], 'x', label='drone')
    plt.plot(original[:,0], original[:,1], '.', label='original')
    plt.show()

    
    ## Find the translation that matches best
    
    def t_min(_x):
        ret = 0.0
        _drone_t = translate_all(drone_r,_x)
        
        for p in _drone_t:
            ## Find closest point in original to p
            i_best = 0
            x_p, y_p = p
            r_best = 9e99
            for i, orig in enumerate(original):
                x_0, y_0 = orig
                r2 = (x_p-x_0)**2 + (y_p-y_0)**2
                if r2 < r_best:
                    r_best = r2
                    i_best = i
                    
            ret += r_best
            
        return ret

    print(t_min([0,0]))
    print(t_min([0.1,0.1]))
    print(t_min([1,1]))
    
    ret = scipy.optimize.minimize(t_min, [0,0], method="BFGS")
    print(ret)

    translate = ret.x
    drone_t = translate_all(drone_r, translate)
    print(f"Translation between drone and original = {translate} degrees")
    # plt.plot(drone[:,0], drone[:,1], 'o', label='drone')
    plt.plot(drone_t[:,0], drone_t[:,1], 'x', label='drone')
    plt.plot(original[:,0], original[:,1], '.', label='original')
    plt.show()

    # Now find the permutation that matches
    p_best = best_permute(drone_t, r_all(original, -angle))
    print(p_best)
    print(original[0])
    
    final = ([list(drone_t[p]) for p in p_best])
    print(final)

    out_json = {"antenna_positions": final}
    with open(ARGS.outfile, "w") as fp:
        json.dump(out_json, fp, indent=4, separators=(",", ": "))

    final = np.array(final)
    # plt.plot(drone[:,0], drone[:,1], 'o', label='drone')
    plt.plot(final[:,0], final[:,1], 'o', label='drone')
    plt.plot(original[:,0], original[:,1], '.', label='original')
    plt.savefig("matched_drone_pos.png")
