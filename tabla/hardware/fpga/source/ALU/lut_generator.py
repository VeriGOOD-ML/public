#!/usr/bin/python

# Amir Yazdanbakhsh
# May 25 - 2014 - version 2.0

# Modified by Chenkai Shao
# September 11, 2016

import sys
import math
from decimal import *
import numpy as np

mu = 0
sigma = 0

def header_fun(input_size, lut_size, function, fraction, input_range):
  header  = ""
  header  = "`timescale 1ns/1ps\n"
  header += "module %s(\n\tin,\n\tout\n);\n\n" % (function)
  header += "\tparameter dataLen = %s;\n" % (input_size)
  header += "\tparameter indexLen = %s;\n" % (int.bit_length(int(lut_size)-1))
  header += "\tparameter fracLen = %s;\n\n" % (fraction)
  #header += "\tparameter inputRange = %s;\n\n" % (input_range)
  header += "\tinput signed [dataLen - 1 : 0] in;\n"
  header += "\toutput reg [dataLen - 1 : 0] out;\n"
  header += "\treg [indexLen - 1 :0] index;\n\n"
  #header += "\treg    [%s:0] mem [0:%s]" % (input_size-1, int(lut_size))

  header += "\talways @(in)\n"
  header += "\tbegin\n"
  header += "\t\tout = 0;\n"

  if (function == "sigmoid"):
      header += "\t\tif (in < -(%s << %s)) begin\n" % ((input_range/ 2), fraction)
      header += "\t\t\tout = 0;\n"
      header += "\t\tend else if (in > (%s << %s)) begin\n" % ((input_range/ 2), fraction)
      header += "\t\t\tout = 1 << %s;\n" % (fraction)
      header += "\t\tend else begin\n"
      input_max   = int(input_size)-1
      lut_max     = int(int.bit_length(int(lut_size)-1)) - 1
      integer_part = int(int.bit_length(int(input_range)-1)) - 1
      header += "\t\tindex[%s]\t= in[%s];\n" % (str(lut_max), str(input_max))
      header += "\t\tindex[%s:%s]\t= in[%s:%s];\n" % (str(lut_max-1), str(lut_max-integer_part), str(int(fraction+integer_part-1)), str(int(fraction)))
      fraction_size = lut_max-integer_part
      header += "\t\tindex[%s:0]\t= in[%s:%s];\n" % (str(fraction_size-1), str(int(fraction)-1), str(int(fraction)-fraction_size))

  elif (function == "gaussian"):
      header += "\t\tif (in < -(%s << %s)) begin\n" % ((input_range/ 2), fraction)
      header += "\t\t\tout = 0;\n"
      header += "\t\tend else if (in > (%s << %s)) begin\n" % ((input_range/ 2), fraction)
      header += "\t\t\tout = 0;\n"
      header += "\t\tend else begin\n"
      input_max   = int(input_size)-1
      lut_max     = int(int.bit_length(int(lut_size)-1)) - 1
      integer_part = int(int.bit_length(int(input_range)-1)) - 1
      header += "\t\tindex[%s]\t= in[%s];\n" % (str(lut_max), str(input_max))
      header += "\t\tindex[%s:%s]\t= in[%s:%s];\n" % (str(lut_max-1), str(lut_max-integer_part), str(int(fraction+integer_part-1)), str(int(fraction)))
      fraction_size = lut_max-integer_part
      header += "\t\tindex[%s:0]\t= in[%s:%s];\n" % (str(fraction_size-1), str(int(fraction)-1), str(int(fraction)-fraction_size))


  header += "\t\tcase(index)\n"
  return header
pass;


def footer_fun(function):

  footer = ""
  footer += "\t\tendcase\n"
  if (function == "sigmoid" or function == "gaussian"):
      footer += "\t\tend\n"
  footer += "\tend\n"
  footer += "endmodule\n"
  return footer

def convert2dec(value_str, tot_bits, frac_bits):
  sign_str=value_str[:1]
  int_str=value_str[1:tot_bits-frac_bits]
  flt_str=value_str[tot_bits-frac_bits:]

  if(sign_str=='1'):
    return -1.0 * float(int(int_str,2) + int(flt_str,2)/float(math.pow(2.0,frac_bits)))
  else:
    return 1.0 * float(int(int_str,2) + int(flt_str,2)/float(math.pow(2.0,frac_bits)))
pass;

def sigmoid(value,steepness):
    return (1.0/float(1.0+math.exp(-2.0*steepness*value)))
pass;

def gaussian(value):
    return np.exp(-np.power(value - mu, 2.) / (2 * np.power(sigma, 2.))) / (math.sqrt(2 * math.pi) * sigma)
pass;

def calculate_function(func, value):
  if(func == "sin"):
    return math.sin(value)
  elif (func == "cos"):
    return math.cos(value)
  elif (func == "sigmoid"):
    return sigmoid(value, 0.5)
  elif (func == "acos"):
    return math.acos(value)
  elif (func == "asin"):
    return math.asin(value)
  elif (func == "gaussian"):
    return gaussian(value)
pass;

def convert2fixed(value, tot_bits, frac_bits):

  (frac_value, int_value)  = math.modf(value)
  flag = 0
  if(abs(int_value) >= pow(2.0, tot_bits-frac_bits-1)):
    int_value = pow(2.0, tot_bits-frac_bits-1)-1

  if(abs(frac_value) > ((math.pow(2.0,frac_bits)-1) / math.pow(2.0,frac_bits))):
    frac_value = (math.pow(2.0,frac_bits)-1) / math.pow(2.0,frac_bits)
    flag = 1;
  else:
    frac_value = (frac_value * math.pow(2.0,frac_bits))

  int_value  = int(int_value)
  if (flag==0):
    frac_value = round(frac_value)
  else:
    #print 'read'
    frac_value = (math.pow(2.0,frac_bits)-1)
  flag = 0

  int_str   = '{0:032b}'.format(int(int_value)) # one bit for sign bit
  frac_str  = '{0:032b}'.format(int(frac_value))

  int_str  = int_str[32-(tot_bits-frac_bits)+1:32]
  frac_str = frac_str[32-(frac_bits):32]

  #print "Integer:            " + int_str
  #print "Fraction:           " + frac_str 

  if(value >=0):
    return "0" + int_str + frac_str
  else:
    return "1" + int_str + frac_str
pass;

def main():

  if(len(sys.argv) < 6):
    print "Usage: lut_generate <# bits> <# fractions> <# entries for interval of [0,1)> <range> <sin|cos|asin|acos|sigmoid>"
    exit(1)

  interval_entries = int(sys.argv[3])
  if (not(((interval_entries & (interval_entries - 1)) == 0) and interval_entries > 0)):  # check if lut_entries is power of 2
    print "<# entries for interval of [0,1)> has to be power of 2"
    exit(1)

  input_range_int = int(sys.argv[4])
  lut_entries = interval_entries * input_range_int
  input_range = float(input_range_int)
  bits = int(sys.argv[1])
  frac = int(sys.argv[2])
  inc  = input_range / lut_entries
  func = sys.argv[5]
  if func == "gaussian":
      global mu
      mu = float(sys.argv[6])
      global sigma
      sigma = float(sys.argv[7])

  print "Generating LUT table for %s function...\n" %(sys.argv[5])
  out_fn = open(sys.argv[5] + ".v", "w")

  data_out = open(sys.argv[5] + ".list", "w")


  out_fn.write(header_fun(bits, lut_entries, func, frac, input_range_int))


  index = lut_entries / 2
  center = 0
  start = center - input_range / 2
  for i in range(int(lut_entries)):
    curr_val  = calculate_function(func, start)
    fixed_val = convert2fixed(curr_val, bits, frac)

    out_fn.write("\t\t\t%s'd%s: out = %s'b%s; // input=%s, output=%s\n" % (str(int.bit_length(int(lut_entries)-1)), str(index), str(bits), fixed_val, str(start), str(curr_val))) 
    data_out.write(fixed_val)
    data_out.write("\n") 
    index += 1
    if (index == lut_entries):
        index = 0
    start += inc
  pass;

  out_fn.write(footer_fun(func))
pass;



if __name__ == "__main__":
  main()
