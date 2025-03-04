use std::iter::Iterator;

pub struct Subsets<'a, E> {
    whole: Vec<&'a E>,
    results: Vec<Vec<&'a E>>,
}

impl<'a, E> Subsets<'a, E> {
    pub fn new<I: Iterator<Item = &'a E>>(iter: I) -> Self {
        let whole = iter.collect();
        let mut res = Self {
            whole,
            results: vec![],
        };
        let mut buf = vec![];
        res.build(&mut buf, 0);
        res
    }

    fn build(&mut self, buf: &mut Vec<&'a E>, idx: usize) {
        if idx == self.whole.len() {
            self.results.push(buf.clone());
            return;
        }

        self.build(buf, idx + 1);

        buf.push(self.whole[idx]);
        self.build(buf, idx + 1);
        buf.pop();
    }
}

impl<'a, E> Iterator for Subsets<'a, E> {
    type Item = Vec<&'a E>;

    fn next(&mut self) -> Option<Self::Item> {
        self.results.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn zero() {
        let subsets: Vec<_> = Subsets::<i32>::new(std::iter::empty()).collect();
        assert_eq!(subsets.len(), 1, "{subsets:?}");
        assert!(subsets[0].is_empty(), "{subsets:?}");
    }

    #[test]
    fn one() {
        let whole = vec![1];
        let subsets: Vec<_> = Subsets::new(whole.iter()).collect();
        check(&whole, &subsets);
    }

    #[test]
    fn two() {
        let whole = vec![1, 2];
        let subsets: Vec<_> = Subsets::new(whole.iter()).collect();
        check(&whole, &subsets);
    }

    fn check(whole: &[i32], subsets: &[Vec<&i32>]) {
        assert_eq!(subsets.len(), 1 << whole.len());
        check_all_are_subset(whole, subsets);
        check_pairwisely_diff(subsets);
    }

    fn check_all_are_subset(whole: &[i32], subsets: &[Vec<&i32>]) {
        let whole: HashSet<i32, ahash::RandomState> = whole.iter().copied().collect();
        for ss in subsets.iter() {
            for e in ss.iter() {
                assert!(whole.contains(e));
            }
        }
    }

    fn check_pairwisely_diff(subsets: &[Vec<&i32>]) {
        let mut iter0 = subsets.iter();
        while let Some(ss0) = iter0.next() {
            for ss1 in iter0.clone() {
                assert_ne!(ss0, ss1);
            }
        }
    }
}
