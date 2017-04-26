# Example implementation of simple arithmetic coding in Python (2.7+).
#
# USAGE 
# 
#  python -i arithmetic.py
#  >>> m = {'a': 1, 'b': 1, 'c': 1}
#  >>> model = dirichlet(m)
#  >>> encode(model, "aabbaacc")	
#  '00011110011110010'
#
# NOTES
#
#  This implementation has many shortcomings, e.g., 
#   - There are several inefficient tests, loops, and conversions
#   - There are a few places where code is uncessarily duplicated
#   - It does not output the coded message as a stream
#   - It can only code short messages due to machine precision
#   - The is no defensive coding against errors (e.g., out-of-model symbols)
#   - I've not implemented a decoder!
#
#  The aim was to make the implementation here as close as possible to
#  the algorithm described in lectures while giving some extra detail about
#  routines such as finding extensions to binary intervals.
#
#  For a more sophisticated implementation, please refer to:
#
#    "Arithmetic Coding for Data Compression"
#    I. H. Witten, R. M. Neal, and J. G. Cleary
#    Communications of the ACM, Col. 30 (6), 1987
#
# AUTHOR: Mark Reid
# CREATED: 2014-09-30

def encode(G, stream):
    '''
    Arithmetically encodes the given stream using the guesser function G
    which returns probabilities over symbols P(x|xs) given a sequence xs.
    '''
    
    u, v = 0.0, 1.0     # The interval [u, v) for the message
    xs, bs = "", ""     # The message xs, and binary code bs
    p = G(xs)           # Compute the initial distribution over symbols

    # Iterate through stream, repeatedly finding the longest binary code 
    # that surrounds the interval for the message so far
    for x in stream:
        # Record the new symbol
        xs += x    

        # Find the interval for the message so far
        F_lo, F_hi = cdf_interval(p, x)        
        u, v = u + (v-u)*F_lo, u + (v-u)*F_hi

        # Find a binary code whose interval surrounds [u,v)
        bs = extend_around(bs, u, v)

        # Update the symbol probabilities
        p = G(xs)

    # Stream finished so find shortest extension of the code that sits inside
    # the top half of [u, v)
    bs = extend_inside(bs, u + (v-u)/2, v)

    return bs

##############################################################################
# Models

def dirichlet(m):
    '''
    Returns a Dirichlet model (as a function) for probabilities with 
    prior counts given by the symbol to count dictionary m.
    Probabilities returned by the returned functions are (symbol, prob)
    dictionaries.
    '''

    # Build a function that returns P(x|xs) based on the priors in m
    # and the counts of the symbols in xs
    def p(xs):
        counts = m.copy()
        for x in xs:
            counts[x] += 1

        total = sum(counts.values())
        return { a: float(c)/total for a, c in counts.items() }

    # Return the constructed function
    return p

##############################################################################
# Interval methods

def cdf_interval(p, a):
    '''
    Compute the cumulative distribution interval [F(a'), F(a)) for the 
    probabilities p (represented as a (symbol,prob) dict) where
    F(a) = P(x <= a) and a' is the symbol preceeding a.
    '''

    F_lo, F_hi = 0, 0

    A = sorted(p)
    for x in A:
        F_lo, F_hi = F_hi, F_hi + p[x]
        if x == a:
            break
    
    return F_lo, F_hi

def binary_interval(bs):
    '''
    Returns an interval [n, m) for n and m integers, and denominator d
    representing the interval [n/d, m/d) for the binary string bs.
    '''
    
    n, d = to_rational(bs)
    return n, n + 1, d

def to_rational(bs):
    '''Return numerator and denominator for ratio of 0.bs.'''
    n = 0
    for b in bs:
        n *= 2
        n += int(b)
    
    return n, 2**len(bs) 

def around(bs, u, v):
    '''Tests whether [0.bs, 0.bs111...) contains [u, v).'''
    n, m, d = binary_interval(bs)
    return (n <= u*d) and (v*d <= m)

def extend_around(bs, u, v):
    '''Find the longest extension of the given binary string so its interval 
       wraps around the interval [u, v).'''
    
    contained = True
    while contained:
        if around(bs + "0", u, v):
            bs += "0"
        elif around(bs + "1", u, v):
            bs += "1"
        else:
            contained = False
    
    return bs

def inside(bs, u, v):
    '''Tests whether [0.bs, 0.bs111...) is contained by [u, v).'''
    n, m, d = binary_interval(bs)
    return (u*d <= n) and (m <= v*d)

def extend_inside(bs, u, v):
    '''Find the shortest extension of the given binary string so its interval 
       sits inside the interval [u, v).'''
    
    while not inside(bs, u, v):
        # Test whether gap between binary interval and [u,v) is bigger at the
        # bottom than at the top
        n, m, d = binary_interval(bs)
        if u*d - n > m - v*d:
            bs += "1"   # If so, move bottom up by halving
        else:
            bs += "0"   # If not, move top down by halving
                
    return bs

