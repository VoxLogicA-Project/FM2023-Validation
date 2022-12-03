.PHONY: archive

archive:
	rm Ciancia-Groote-Latella-Massink-de_Vink-FM2023.zip
	git archive --format zip --output Ciancia-Groote-Latella-Massink-de_Vink-FM2023.zip main
