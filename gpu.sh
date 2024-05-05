if [ ! -e /etc/apt/sources.list.d/nvidia.list ]; then
  echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/nvidia.list
fi

if [ ! -e /etc/apt/trusted.gpg.d/nvidia.asc ]; then
cat << EOF > /etc/apt/trusted.gpg.d/nvidia.asc
-----BEGIN PGP PUBLIC KEY BLOCK-----
Version: GnuPG v2.0.22 (GNU/Linux)

mQINBGJYmlEBEAC6nJmeqByeReM+MSy4palACCnfOg4pOxffrrkldxz4jrDOZNK4
q8KG+ZbXrkdP0e9qTFRvZzN+A6Jw3ySfoiKXRBw5l2Zp81AYkghV641OpWNjZOyL
syKEtST9LR1ttHv1ZI71pj8NVG/EnpimZPOblEJ1OpibJJCXLrbn+qcJ8JNuGTSK
6v2aLBmhR8VR/aSJpmkg7fFjcGklweTI8+Ibj72HuY9JRD/+dtUoSh7z037mWo56
ee02lPFRD0pHOEAlLSXxFO/SDqRVMhcgHk0a8roCF+9h5Ni7ZUyxlGK/uHkqN7ED
/U/ATpGKgvk4t23eTpdRC8FXAlBZQyf/xnhQXsyF/z7+RV5CL0o1zk1LKgo+5K32
5ka5uZb6JSIrEPUaCPEMXu6EEY8zSFnCrRS/Vjkfvc9ViYZWzJ387WTjAhMdS7wd
PmdDWw2ASGUP4FrfCireSZiFX+ZAOspKpZdh0P5iR5XSx14XDt3jNK2EQQboaJAD
uqksItatOEYNu4JsCbc24roJvJtGhpjTnq1/dyoy6K433afU0DS2ZPLthLpGqeyK
MKNY7a2WjxhRmCSu5Zok/fGKcO62XF8a3eSj4NzCRv8LM6mG1Oekz6Zz+tdxHg19
ufHO0et7AKE5q+5VjE438Xpl4UWbM/Voj6VPJ9uzywDcnZXpeOqeTQh2pQARAQAB
tCBjdWRhdG9vbHMgPGN1ZGF0b29sc0BudmlkaWEuY29tPokCOQQTAQIAIwUCYlia
UQIbAwcLCQgHAwIBBhUIAgkKCwQWAgMBAh4BAheAAAoJEKS0aZY7+GPM1y4QALKh
BqSozrYbe341Qu7SyxHQgjRCGi4YhI3bHCMj5F6vEOHnwiFH6YmFkxCYtqcGjca6
iw7cCYMow/hgKLAPwkwSJ84EYpGLWx62+20rMM4OuZwauSUcY/kE2WgnQ74zbh3+
MHs56zntJFfJ9G+NYidvwDWeZn5HIzR4CtxaxRgpiykg0s3ps6X0U+vuVcLnutBF
7r81astvlVQERFbce/6KqHK+yj843Qrhb3JEolUoOETK06nD25bVtnAxe0QEyA90
9MpRNLfR6BdjPpxqhphDcMOhJfyubAroQUxG/7S+Yw+mtEqHrL/dz9iEYqodYiSo
zfi0b+HFI59sRkTfOBDBwb3kcARExwnvLJmqijiVqWkoJ3H67oA0XJN2nelucw+A
Hb+Jt9BWjyzKWlLFDnVHdGicyRJ0I8yqi32w8hGeXmu3tU58VWJrkXEXadBftmci
pemb6oZ/r5SCkW6kxr2PsNWcJoebUdynyOQGbVwpMtJAnjOYp0ObKOANbcIg+tsi
kyCIO5TiY3ADbBDPCeZK8xdcugXoW5WFwACGC0z+Cn0mtw8z3VGIPAMSCYmLusgW
t2+EpikwrP2inNp5Pc+YdczRAsa4s30Jpyv/UHEG5P9GKnvofaxJgnU56lJIRPzF
iCUGy6cVI0Fq777X/ME1K6A/bzZ4vRYNx8rUmVE5
=DO7z
-----END PGP PUBLIC KEY BLOCK-----
EOF
fi

apt-get update
apt install python-is-python3 python3-pip dkms cuda-12-2 nvtop -uy
apt-mark hold $(dpkg -l |grep  'libcudnn\|nvidia-\|cuda-\|xserver-xorg-video-nvidia-\|libnvidia-'|awk '{print $2}')
# https://www.tensorflow.org/install/pip
#    python3 -m pip install tensorflow
# https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio
