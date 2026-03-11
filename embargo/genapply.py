################################################################################
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import os
import sys

script_dir = os.path.abspath(os.path.dirname(__file__))

class Diff:
    def __init__(self, lines):
        self.lines = lines

    def dump(self):
        return '\nDIFF:\n  ' + '  '.join(self.lines)

    def path(self):
        for l in self.lines:
            if l.startswith('+++ b'):
                return '/'.join(l.split('/')[1:]).strip()
        assert False, f'Cannot parse path: {self.dump()}'

    def is_new(self):
        return self.lines[1].startswith('new file')

    def new_lines(self):
        return [l[1:] for l in self.lines[5:] if l.startswith('+')]

work_dir = sys.argv[1]
lines = open(sys.argv[2]).readlines()
diffs = []
cur = []
for l in lines:
    if l.startswith('diff --git'):
        if cur: diffs.append(Diff(cur))
        cur = [l]
    else:
        cur.append(l)

if cur: diffs.append(Diff(cur))

for diff in diffs:
    path = diff.path()
    if not diff.is_new(): continue
    print(f'Writing a new file: {path}')
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    open(path, 'w').write(''.join(diff.new_lines()))

apply_lines = []
for diff in diffs:
    if diff.is_new(): continue
    s_diff = ''.join(diff.lines)
    diff_tmp_path = work_dir + '/' + diff.path().replace('/', '_') + '.diff'
    open(diff_tmp_path, 'w').write(s_diff)
    apply_lines.append(f'git apply --3way {diff_tmp_path} |& tee -a {work_dir}/merge.log')

open(f'{work_dir}/apply.sh', 'w').write('\n'.join(apply_lines))
