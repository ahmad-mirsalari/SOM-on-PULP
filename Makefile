PULP_APP = som

PULP_APP_FC_SRCS = main.c
PULP_APP_SRCS =  som.c

PULP_CFLAGS += -O3 -g3 
PULP_CFLAGS += -mno-memcpy


ifdef cores
PULP_CFLAGS += -DNUM_CORES=${cores} #-flto -DFABRIC=1
else
PULP_CFLAGS += -DNUM_CORES=1
endif


ifdef FABRIC
PULP_CFLAGS += -DFABRIC
endif

ifdef cores
PULP_CFLAGS += -DUSE_INTRINSICS
endif

PULP_CFLAGS += -fno-tree-vectorize

# FP FORMAT
ifdef fmt
PULP_CFLAGS += -D${fmt}
else
PULP_CFLAGS += -DFP32
endif

# VECTORIAL FORMAT for half-precision FP
ifdef vec
PULP_CFLAGS += -DVECTORIAL
endif

# CHECK RESULTS
ifdef check
PULP_CFLAGS += -DCHECK
endif

# PRINT RESULTS
ifdef PRINT_RESULTS
PULP_CFLAGS += -DPRINT_RESULTS
endif

ifdef verbose
PULP_CFLAGS += -DVERBOSE
endif

# STATISTICS
ifdef stats
PULP_CFLAGS += -DSTATS
endif

ifdef w_block
PULP_CFLAGS += -Dw_b=${w_block}
endif

ifdef i_block
PULP_CFLAGS += -Dinp_b=${i_block}
endif

ifdef IN_ORDER
PULP_CFLAGS += -DIN_ORDER
endif

include $(RULES_DIR)/pmsis_rules.mk
