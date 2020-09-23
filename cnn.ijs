require 'format/printf'
require 'stats/base'

conv          =: 4 : '($x) ([: +/@:, x&*);._3 y'
multiconv     =: 3 : 0
  'a ws bias' =. y
  bias ([ + a conv~ ])"(0,#$a) ws
)
backbias      =: +/@:,
backin        =: 3 : 0
  'd w in' =. y
  +/ (- (#: i.@:(*/)) $w) |."(1,#$in) ($in) {."(#$in) (,w) */ d
)
logistic      =: %@:>:@:^@:-
meansqerr     =: -:@:(+/)@:,@:*:@:-
backlogistic  =: * * 1 - ]
avgpool       =: 4 %~ (2 2 $ 2)&(+/@:,;._3)"2
backavgpool   =: (2 (#"1) 2 # ])"2
backmulticonv =: 3 : 0
  'd_out weights in bias' =. y
  d_in   =. +/ d_out ([: backin ; , <@in)"(#$in) weights
  d_w    =. conv&in"(#$in) d_out
  d_bias =. backbias"(#$in) d_out
  d_in ; d_w ; d_bias
)

trainzhang =: 3 : 0
  'img target k1 b1 k2 b2 fc b' =. y
  c1    =. logistic multiconv img ; k1 ; b1
  s1    =. avgpool c1
  c2    =. logistic multiconv s1 ; k2 ; b2
  s2    =. avgpool c2
  out   =. logistic multiconv s2 ; fc ; b
  d_out =. out - target
  err   =. out meansqerr target

  'd_s2 d_fc d_b'  =. backmulticonv (d_out backlogistic out) ; fc ; s2 ; b
  d_c2             =. backavgpool d_s2
  bl1              =. d_c2 backlogistic c2
  'd_s1 d_k2 d_b2' =. backmulticonv bl1 ; k2 ; s1 ; b2
  d_c1             =. backavgpool d_s1
  'd_k1 d_b1'      =. }. backmulticonv (d_c1 backlogistic c1) ; k1 ; img ; b1
  d_k1 ; d_b1 ; d_k2 ; d_b2 ; d_fc ; d_b ; err
)

testzhang =: 4 : 0
  'k1 b1 k2 b2 fc b' =. y
  c1  =. logistic multiconv x ; k1 ; b1
  s1  =. avgpool c1
  c2  =. logistic multiconv s1 ; k2 ; b2
  s2  =. avgpool c2
  out =. logistic multiconv s2 ; fc ; b
  (i. >./) out
)

train =: 3 : 0
  t =. 6!:1 ''
  'k1 b1 k2 b2 fc b rate imgs labs trsz' =. y
  e =. 0
  i =. 0

  shuf =. ?~ trsz
  imgs =. shuf { imgs
  labs =. shuf { labs

  while. i < trsz do.
    img =. i { imgs
    target =. 10 1 1 1 1 $ (i. 10) = i { labs
    'd_k1 d_b1 d_k2 d_b2 d_fc d_b err' =. trainzhang img;target;k1;b1;k2;b2;fc;b
    k1 =. k1 - rate * d_k1
    k2 =. k2 - rate * d_k2
    b1 =. b1 - rate * d_b1
    b2 =. b2 - rate * d_b2
    fc =. fc - rate * d_fc
    b  =. b  - rate * d_b
    e  =. e + +/ err
    i  =. i + 1
  end.
  'Training took: %.2f seconds' printf (6!:1 '') - t
  'Average error: %.3f' printf e % trsz
  k1 ; b1 ; k2 ; b2 ; fc ; b
)

readimages =: 3 : 0
  t =. a. i. fread y
  z =. (_4 (256&#.)\ 12 {. 4 }. t) $ 16 }. t
  'Read %d images from %s' printf (#z) ; y
  z
)
readlabels =: 3 : 0
  z =. 8 }. a. i. fread y
  'Read %d labels from %s' printf (#z) ; y
  z
)

main =: 3 : 0
  epochs    =. 10
  trainings =. 1000
  tests     =. 10000
  rate      =. 0.05

  init =. 4 : '(%: 6 % x) * <: +: ? y $ 0'
  k1   =. (25 * 1 + 6) init 6 5 5
  b1   =. 6 $ 0
  k2   =. (25 * 6 + 12) init 12 6 5 5
  b2   =. 12 $ 0
  fc   =. (192 + 10) init 10 12 1 4 4
  b    =. 10 $ 0

  trimgs =. readimages 'input/train-images-idx3-ubyte'
  trlabs =. readlabels 'input/train-labels-idx1-ubyte'
  teimgs =. readimages 'input/t10k-images-idx3-ubyte'
  telabs =. readlabels 'input/t10k-labels-idx1-ubyte'

  trmean =. mean trainings {. trimgs
  trstd  =. (+ 0&=) stddev trainings {. trimgs
  trimgs =. (trimgs -"2 trmean) %"2 trstd
  teimgs =. (teimgs -"2 trmean) %"2 trstd

  'Running Zhang with %d epochs' printf epochs
  '%d training images, %d tests, and a rate of %.3f' printf trainings ; tests ; rate

  'k1 b1 k2 b2 fc b' =. ([: train ,&(rate;trimgs;trlabs;trainings))^:epochs k1;b1;k2;b2;fc;b

  t =. 6!:1 ''
  correct =. +/ telabs = teimgs testzhang"2 k1;b1;k2;b2;fc;b
  'Recognition took %.2f seconds' printf (6!:1 '') - t
  '%d images out of %d recognized correctly' printf correct ; tests
)

9!:1 ] 16807
main ''
exit ''
