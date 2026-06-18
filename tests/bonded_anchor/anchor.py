import numpy as np, subprocess, os, math
D="/tmp/anchor"; os.makedirs(D, exist_ok=True)
FASTMD=os.path.expanduser("~/fastMD/build/fastmd")

def write(name, text): open(os.path.join(D,name),"w").write(text)
def run(conf):
    r=subprocess.run([FASTMD,conf],cwd=D,capture_output=True,text=True)
    if r.returncode!=0: print("FASTMD FAIL",conf,"\n",r.stderr[-800:]); raise SystemExit(1)
def forces(fn):
    rows=[l.split() for l in open(os.path.join(D,fn)) if l.strip() and not l.startswith("#")]
    return np.array([[float(x) for x in r[1:4]] for r in rows])

# ---------- ANGLE anchor: table vs analytic ----------
k, th0_deg = 5.0, 110.0; th0=math.radians(th0_deg)
th=np.linspace(0,math.pi,3601); U=k*(th-th0)**2; f=-2*k*(th-th0)   # f=-dU/dtheta
lines=["# angle tab","","angle",f"N {len(th)}",""]+[f"{i+1} {th[i]:.10g} {U[i]:.10g} {f[i]:.10g}" for i in range(len(th))]
write("angle.tab","\n".join(lines)+"\n")
write("angle.data","""LAMMPS

3 atoms
1 atom types
1 angles
1 angle types

0.0 20.0 xlo xhi
0.0 20.0 ylo yhi
0.0 20.0 zlo zhi

Masses

1 1.0

Atoms # molecular

1 1 1 1.0 0.0 0.0
2 1 1 0.0 0.0 0.0
3 1 1 0.0 1.0 0.0

Angles

1 1 1 2 3
""")
base="natoms 3\nntypes 1\nrc 2.0\nskin 0.3\ndt 0.001\nnsteps 0\ndump_freq 1\nnvt_langevin 1.0 1.0 1.0 42\nlammps_data_file angle.data\n"
write("ang_h.conf", base+"angle_type 0 %g %g\nforces_dump_file fa_h.txt\n"%(k,th0_deg))
write("ang_t.conf", base+"angle_table 0 angle.tab angle\nforces_dump_file fa_t.txt\n")
run("ang_h.conf"); run("ang_t.conf")
fh, ft = forces("fa_h.txt"), forces("fa_t.txt")
ang_err=np.max(np.abs(fh-ft))
print("ANGLE harmonic forces:\n", fh)
print("ANGLE table forces:\n", ft)
print(f"ANGLE max|table-analytic| = {ang_err:.3e}")

# ---------- BOND anchor: table vs hand-computed ----------
kb, req = 10.0, 1.0
rb=np.linspace(0.5,2.5,401); Ub=0.5*kb*(rb-req)**2
with np.errstate(divide="ignore",invalid="ignore"):
    fb=np.where(rb>0, (-kb*(rb-req))/rb, 0.0)   # packed F/r
lines=["# bond tab","","bond",f"N {len(rb)}",""]+[f"{i+1} {rb[i]:.10g} {Ub[i]:.10g} {fb[i]:.10g}" for i in range(len(rb))]
write("bond.tab","\n".join(lines)+"\n")
write("bond.data","""LAMMPS

2 atoms
1 atom types
1 bonds
1 bond types

0.0 20.0 xlo xhi
0.0 20.0 ylo yhi
0.0 20.0 zlo zhi

Masses

1 1.0

Atoms # molecular

1 1 1 0.0 0.0 0.0
2 1 1 1.5 0.0 0.0

Bonds

1 1 1 2
""")
write("bond_t.conf","natoms 2\nntypes 1\nrc 2.0\nskin 0.3\ndt 0.001\nnsteps 0\ndump_freq 1\nnvt_langevin 1.0 1.0 1.0 42\nlammps_data_file bond.data\nbond_table 0 bond.tab bond\nforces_dump_file fb_t.txt\n")
run("bond_t.conf")
fb_out=forces("fb_t.txt")
# expected: |F|=k(r-req)=10*0.5=5; atom1 +x, atom2 -x
expected=np.array([[5.0,0,0],[-5.0,0,0]])
bond_err=np.max(np.abs(fb_out-expected))
print("BOND table forces:\n", fb_out)
print("BOND expected:\n", expected)
print(f"BOND max|table-handcalc| = {bond_err:.3e}")

ok = ang_err<1e-3 and bond_err<1e-3
print("ANCHOR RESULT:", "PASS" if ok else "FAIL")
