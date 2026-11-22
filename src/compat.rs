//! Smooths over what differs between the `ash` release (0.38) and the git version that the
//! `ash_latest` feature builds against.

/// `T::new(instance, device)` in the release, `T::load(instance, device)` in git.
macro_rules! load_ext {
    ($ext:ty, $instance:expr, $device:expr) => {{
        #[cfg(not(feature = "ash_latest"))]
        let ext = <$ext>::new($instance, $device);
        #[cfg(feature = "ash_latest")]
        let ext = <$ext>::load($instance, $device);
        ext
    }};
}
pub(crate) use load_ext;

/// The 0.38's `push_next`, which ash latest renamed to the unsafe `extend`.
#[cfg(feature = "ash_latest")]
pub(crate) trait PushNext<'a>: crate::ash::vk::TaggedStructure<'a> {
    fn push_next<'b: 'a, T: crate::ash::vk::Extends<Self> + crate::ash::vk::TaggedStructure<'b>>(
        self,
        next: &'a mut T,
    ) -> Self {
        // SAFETY: every `p_next` in the chain of `next` is null or points to a live struct, as it
        // did for the release's `push_next`.
        unsafe { self.extend(next) }
    }
}
#[cfg(feature = "ash_latest")]
impl<'a, S: crate::ash::vk::TaggedStructure<'a>> PushNext<'a> for S {}
