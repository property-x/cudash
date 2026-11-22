//! `push_next` as in ash 0.38, which ash latest renamed to the unsafe `extend`.

#[cfg(feature = "ash_latest")]
pub trait PushNext<'a>: cudash::ash::vk::TaggedStructure<'a> {
    fn push_next<
        'b: 'a,
        T: cudash::ash::vk::Extends<Self> + cudash::ash::vk::TaggedStructure<'b>,
    >(
        self,
        next: &'a mut T,
    ) -> Self {
        // SAFETY: the tests only chain fresh default structs.
        unsafe { self.extend(next) }
    }
}
#[cfg(feature = "ash_latest")]
impl<'a, S: cudash::ash::vk::TaggedStructure<'a>> PushNext<'a> for S {}
