# Create formal specification document
echo "# Calvin Intelligence Specification" > SPECIFICATION.md
echo "## Version 1.0 $(date +%Y-%m-%d)" >> SPECIFICATION.md
echo "$$I_{\text{meta}} = \oint\limits_{\Delta} \frac{\delta R \otimes \delta B \otimes \delta G}{\epsilon}$$" >> SPECIFICATION.md
git add SPECIFICATION.md
git commit -m "Added formal specification v1.0"