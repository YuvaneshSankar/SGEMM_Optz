.PHONY: bank clean




GCC = gcc
NVCC = nvcc



bank:
	@$(NVCC) 01_bank_conflicts.cu -o  01_bank_conflicts
	@./01_bank_conflicts


clean:
	@rm -f 01_bank_conflicts