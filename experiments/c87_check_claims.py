# C-87: my own check of the identities and the Adam toy example in the returned note (nothing is taken from their files)
import numpy as np
n = 8; idx = np.arange(1 << 16); a = idx & 255; b = idx >> 8
p4 = ((a ^ b) >> 4) & 1; c4 = (((a & 15) + (b & 15)) >> 4) & 1; s4 = ((a + b) >> 4) & 1
chk = lambda v: 1 - 2 * v
S, P, C = chk(s4), chk(p4), chk(c4)
print("S == P*C on all", len(idx), "pairs:", bool((S == P * C).all()))
print("P(c4=1) = %.5f (15/32 = %.5f)" % (c4.mean(), 15 / 32)); print("E[C] = %.5f (1/16 = %.5f); E[S P] = %.5f; E[P P] = %.1f" % (C.mean(), 1 / 16, (S * P).mean(), (P * P).mean()))
U3 = chk((a >> 3) & 1); print("E[S P U3] with U3 = (-1)^a3: %.5f ; with (-1)^b3: %.5f" % ((S * P * U3).mean(), (S * P * chk((b >> 3) & 1)).mean()))
print("E[S U3] (without P): %.5f ; E[S a4-char] etc:" % ((S * U3).mean()), "E[S (-1)^a4]: %.5f" % ((S * chk((a >> 4) & 1)).mean()))
d = 0.3; g1 = np.array([-d, -d]); g2 = np.array([-1.0, 1.0]); eta = 1.0
solo = -eta * np.sign(g1); joint = -eta * np.sign(g1 + g2)
print("g1.g2 =", g1 @ g2, "| solo step", solo, "joint step", joint, "| first-order change of loss 1: solo %.2f joint %.2f" % (g1 @ solo, g1 @ joint))
