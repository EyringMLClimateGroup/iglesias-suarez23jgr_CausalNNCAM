import sys, getopt

def main(argv):
   spcam_parents  = ['tbp','qbp','vbp','ps','solin','shflx','lhflx']
   spcam_children = ['tphystnd','prect', 'fsns', 'flns']
   region         = None
   lim_levels     = None
   target_levels  = None
   list_pc_alpha  = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
   try:
      opts, args = getopt.getopt(argv,"hp:c:r:l:t:a",["parents=","children=",
                                                      "region=","lim_levels=",
                                                      "target_levels=","pc-alpha"])
   except getopt.GetoptError:
      print ('pipeline.py -p [parents] -c [children] -r [region] -l [lim_levels] -t [target_levels] -a [pc-alpha]')
      sys.exit(2)
   for opt, arg in opts:
      print(opt, arg)
      if opt == '-h':
         print ('pipeline.py -p [parents] -c [children]')
         sys.exit()
      elif opt in ("-p", "--parents"):
         spcam_parents = arg
      elif opt in ("-c", "--children"):
         spcam_children = arg
      elif opt in ("-r", "--region"):
         region = arg
      elif opt in ("-l", "--lim_levels"):
         lim_levels = arg
      elif opt in ("-t", "--target_levels"):
         target_levels = arg
      elif opt in ("-a", "--pc-alpha"):
         list_pc_alpha = arg
   print ('Parents are: ', spcam_parents)
   print ('Children are: ', spcam_children)
   print ('Region is: ', region)
   print ('lim_levels are: ', lim_levels)
   print ('target_levels are: ', target_levels)
   print ('pc-alpha is: ', list_pc_alpha)

if __name__ == "__main__":
   main(sys.argv[1:])
